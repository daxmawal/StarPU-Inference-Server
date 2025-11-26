#pragma once

namespace starpu_server {

struct RuntimeConfig;

void run_trace_plots_if_enabled(const RuntimeConfig& opts);

}  // namespace starpu_server
