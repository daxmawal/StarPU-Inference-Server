#pragma once
#include "args_parser.hpp"
#include "starpu_setup.hpp"

void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu);