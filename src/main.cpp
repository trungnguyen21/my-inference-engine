#include "app.h"

int main(int argc, char **argv)
{
    AppConfig config;
    app_config_init(&config);

    int parse_result = app_parse_args(argc, argv, &config);
    if (parse_result != 0) {
        return parse_result > 0 ? 0 : 2;
    }

    return app_run(&config);
}
