// Command loadgen drives continuous traffic at the inference server and shows
// a live tally of which pod and node answered.
//
// The point is to make disruption legible. Deleting a pod on an idle cluster
// shows nothing; with traffic flowing you watch one bar stop growing while the
// others absorb the load, then a new pod appear and start climbing.
//
//	go run ./loadgen -url http://localhost:8080 -users 6
//
// Two deliberate choices:
//
//   - Short requests. Each takes 2-5 s, so a large max-tokens would show three
//     completions in a whole demo. This tool answers "where did requests go",
//     not "how fast is generation" — bench.sh answers that.
//   - 503 is retried and counted apart from real errors. A restarting pod
//     returns "model is still loading" and a saturated one returns "all N
//     workers busy". Treating those as failures makes resilience look like an
//     outage; hiding them entirely would be dishonest. So: counted, shown,
//     retried.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

type request struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
}

type response struct {
	Text            string `json:"text"`
	GeneratedTokens int    `json:"generated_tokens"`
	Pod             string `json:"pod"`
	Node            string `json:"node"`
}

// result is one completed attempt.
type result struct {
	pod, node string
	latency   time.Duration
	tokens    int
	outcome   outcome
	err       string
}

type outcome int

const (
	ok      outcome = iota
	busy            // 503: saturated, or still loading. Expected during a restart.
	failed          // anything else
	aborted         // cancelled by our own shutdown; not a server failure
)

// tally accumulates results. Guarded by a mutex because the virtual users all
// write to it; the renderer reads a snapshot.
type tally struct {
	mu sync.Mutex

	start              time.Time
	okCount, busyCount int
	failCount          int
	tokens             int
	perPod             map[podKey]*podStat
	lastErrs           []string
}

type podKey struct{ node, pod string }

type podStat struct {
	count     int
	busy      int
	failed    int
	tokens    int
	latencies []time.Duration
	lastSeen  time.Time
}

func newTally() *tally {
	return &tally{start: time.Now(), perPod: map[podKey]*podStat{}}
}

func (t *tally) add(r result) {
	if r.outcome == aborted {
		return // our shutdown, not a result
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	switch r.outcome {
	case ok:
		t.okCount++
		t.tokens += r.tokens
	case busy:
		t.busyCount++
	case failed:
		t.failCount++
		if r.err != "" {
			t.lastErrs = append(t.lastErrs, fmt.Sprintf("%s  %s",
				time.Now().Format("15:04:05"), r.err))
			if len(t.lastErrs) > 4 {
				t.lastErrs = t.lastErrs[len(t.lastErrs)-4:]
			}
		}
	}

	// An unidentified responder still deserves a row; that itself is a signal
	// that POD_NAME is not being set.
	if r.pod == "" && r.outcome != failed {
		r.pod = "unknown"
	}
	if r.pod == "" {
		return
	}
	key := podKey{node: r.node, pod: r.pod}
	st := t.perPod[key]
	if st == nil {
		st = &podStat{}
		t.perPod[key] = st
	}
	st.lastSeen = time.Now()
	switch r.outcome {
	case ok:
		st.count++
		st.tokens += r.tokens
		st.latencies = append(st.latencies, r.latency)
	case busy:
		st.busy++
	case failed:
		st.failed++
	}
}

func median(ds []time.Duration) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return cp[len(cp)/2]
}

// render draws the whole screen. Redrawing in place rather than scrolling is
// what makes a bar stopping mid-demo obvious.
func (t *tally) render(target string, users int) string {
	t.mu.Lock()
	defer t.mu.Unlock()

	elapsed := time.Since(t.start)
	rps := float64(t.okCount) / elapsed.Seconds()
	tps := float64(t.tokens) / elapsed.Seconds()

	var b strings.Builder
	fmt.Fprintf(&b, "\033[H\033[2J") // home, clear
	fmt.Fprintf(&b, "loadgen → %s   %d concurrent\n", target, users)
	fmt.Fprintf(&b, "%s\n", strings.Repeat("─", 72))
	fmt.Fprintf(&b, "elapsed %-9s  req/s %-6.2f  tok/s %-7.1f  ok %-5d busy %-5d err %d\n\n",
		elapsed.Round(time.Second), rps, tps, t.okCount, t.busyCount, t.failCount)

	if len(t.perPod) == 0 {
		b.WriteString("  (no responses yet)\n")
		return b.String()
	}

	// Group by node so a two-node cluster reads as two machines, not a list.
	byNode := map[string][]podKey{}
	for k := range t.perPod {
		byNode[k.node] = append(byNode[k.node], k)
	}
	nodes := make([]string, 0, len(byNode))
	for n := range byNode {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	// One scale across every bar, so bar lengths are comparable.
	maxCount := 1
	for _, st := range t.perPod {
		if st.count > maxCount {
			maxCount = st.count
		}
	}

	for _, node := range nodes {
		label := node
		if label == "" {
			label = "(no node name)"
		}
		nodeTotal := 0
		for _, k := range byNode[node] {
			nodeTotal += t.perPod[k].count
		}
		fmt.Fprintf(&b, "  %s  (%d requests)\n", label, nodeTotal)

		keys := byNode[node]
		sort.Slice(keys, func(i, j int) bool { return keys[i].pod < keys[j].pod })
		for _, k := range keys {
			st := t.perPod[k]
			bar := strings.Repeat("█", st.count*28/maxCount)
			// A pod that has not answered recently is probably gone; showing it
			// greyed rather than dropping it is what makes a deletion visible.
			marker := " "
			if time.Since(st.lastSeen) > 15*time.Second {
				marker = "·"
			}
			fmt.Fprintf(&b, "   %s %-22s %-28s %4d", marker, truncate(k.pod, 22), bar, st.count)
			if p50 := median(st.latencies); p50 > 0 {
				fmt.Fprintf(&b, "  p50 %s", p50.Round(10*time.Millisecond))
			}
			if st.busy > 0 {
				fmt.Fprintf(&b, "  busy %d", st.busy)
			}
			if st.failed > 0 {
				fmt.Fprintf(&b, "  err %d", st.failed)
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	if len(t.lastErrs) > 0 {
		b.WriteString("  recent errors\n")
		for _, e := range t.lastErrs {
			fmt.Fprintf(&b, "    %s\n", truncate(e, 66))
		}
		b.WriteString("\n")
	}
	b.WriteString("  ctrl-c to stop\n")
	return b.String()
}

func (t *tally) summary() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	elapsed := time.Since(t.start)

	var b strings.Builder
	fmt.Fprintf(&b, "\nran %s: %d ok, %d busy (retried), %d failed, %d tokens\n",
		elapsed.Round(time.Second), t.okCount, t.busyCount, t.failCount, t.tokens)
	if t.okCount > 0 {
		fmt.Fprintf(&b, "%.2f req/s, %.1f tok/s aggregate\n",
			float64(t.okCount)/elapsed.Seconds(), float64(t.tokens)/elapsed.Seconds())
	}
	fmt.Fprintf(&b, "\n%-16s %-24s %8s %8s\n", "NODE", "POD", "REQUESTS", "P50")
	keys := make([]podKey, 0, len(t.perPod))
	for k := range t.perPod {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].node != keys[j].node {
			return keys[i].node < keys[j].node
		}
		return keys[i].pod < keys[j].pod
	})
	for _, k := range keys {
		st := t.perPod[k]
		fmt.Fprintf(&b, "%-16s %-24s %8d %8s\n",
			orDash(k.node), truncate(k.pod, 24), st.count,
			median(st.latencies).Round(10*time.Millisecond))
	}
	return b.String()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

var prompts = []string{
	"Generate tests that measure response time differences between cached and uncached requests for GET /api/content/:id. Clear the cache, measure cold response time, then measure warm response time. Assert that cache hit responses are at least 5x faster than cache misses. Report cache hit ratio over 1000 requests.",
	"Write a performance test that simulates API requests from five geographic regions (US-East, US-West, EU-West, Asia-Pacific, South America). Compare response times across regions for the /api/search endpoint. Flag any region where latency exceeds 2x the lowest-region baseline. Output a latency risk heatmap by region.",
	"Write a stress test that ramps from 50 to 2000 concurrent users in 60 seconds on POST /api/orders. Monitor throughput, error rates, and response times during the spike. Include assertions that verify the system degrades gracefully rather than returning 500 errors. Report the exact concurrency level where error rate exceeds 5%.",
	"What color is the sky?",
	"Who wrote Hamlet?",
}

// user runs one virtual client: request, record, repeat, until ctx ends.
func user(ctx context.Context, id int, target string, maxTokens int, t *tally, client *http.Client) {
	for i := 0; ; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		body, _ := json.Marshal(request{
			Prompt:    prompts[(id+i)%len(prompts)],
			MaxTokens: maxTokens,
		})
		start := time.Now()
		r := doOne(ctx, client, target, body)
		r.latency = time.Since(start)
		t.add(r)

		// Back off before retrying. Without this, a server that is down turns
		// this loop into a tight spin — measured at 311k connection errors in
		// 20 seconds, which floods the tally and burns CPU.
		//
		// 503 is a short wait (a pod is restarting, seconds away). A hard
		// failure waits longer, since the server may be gone entirely.
		var wait time.Duration
		switch r.outcome {
		case busy:
			wait = 500 * time.Millisecond
		case failed:
			wait = 2 * time.Second
		}
		if wait > 0 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(wait):
			}
		}
	}
}

func doOne(ctx context.Context, client *http.Client, target string, body []byte) result {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		target+"/generate", bytes.NewReader(body))
	if err != nil {
		return result{outcome: failed, err: err.Error()}
	}
	if ctx.Err() != nil {
		return result{outcome: aborted}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		// Our own ctrl-c or -duration cancels in-flight requests. Counting
		// those as failures would end every run with a misleading error count.
		if ctx.Err() != nil {
			return result{outcome: aborted}
		}
		return result{outcome: failed, err: err.Error()}
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)

	// Identity comes from the header so it is available even on an error,
	// which is how a refusing pod still shows up in the tally.
	pod, node := "", ""
	if sb := resp.Header.Get("X-Served-By"); sb != "" {
		if i := strings.IndexByte(sb, '/'); i >= 0 {
			pod, node = sb[:i], sb[i+1:]
		} else {
			pod = sb
		}
	}

	switch {
	case resp.StatusCode == http.StatusServiceUnavailable:
		return result{pod: pod, node: node, outcome: busy}
	case resp.StatusCode != http.StatusOK:
		return result{pod: pod, node: node, outcome: failed,
			err: fmt.Sprintf("HTTP %d: %s", resp.StatusCode, truncate(strings.TrimSpace(string(raw)), 50))}
	}

	var parsed response
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return result{pod: pod, node: node, outcome: failed, err: "bad JSON: " + err.Error()}
	}
	if parsed.Pod != "" {
		pod, node = parsed.Pod, parsed.Node
	}
	return result{pod: pod, node: node, tokens: parsed.GeneratedTokens, outcome: ok}
}

func main() {
	target := flag.String("url", "http://localhost:8080", "base URL of the server")
	users := flag.Int("users", 6, "concurrent requests kept in flight")
	maxTokens := flag.Int("max-tokens", 8,
		"tokens per request; keep this small so requests cycle fast")
	duration := flag.Duration("duration", 0, "stop after this long (0 = until ctrl-c)")
	refresh := flag.Duration("refresh", 500*time.Millisecond, "screen redraw interval")
	timeout := flag.Duration("timeout", 2*time.Minute, "per-request timeout")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *duration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *duration)
		defer cancel()
	}

	client := &http.Client{
		Timeout: *timeout,
		// Each virtual user needs its own connection, or they serialize on a
		// shared one and the tool measures itself rather than the server.
		Transport: &http.Transport{
			MaxIdleConnsPerHost: *users + 2,
			MaxConnsPerHost:     *users + 2,
		},
	}

	t := newTally()
	var wg sync.WaitGroup
	for i := 0; i < *users; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			user(ctx, id, strings.TrimRight(*target, "/"), *maxTokens, t, client)
		}(i)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(*refresh)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				fmt.Print(t.render(*target, *users))
			}
		}
	}()

	wg.Wait()
	<-done
	fmt.Print("\033[H\033[2J")
	fmt.Print(t.summary())
}
