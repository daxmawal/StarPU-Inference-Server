#ifndef STARPU_SETUP_HPP
#define STARPU_SETUP_HPP

#include <starpu.h>
#include <iostream>

class StarPUSetup
{
 public:
  StarPUSetup(const char* sched_policy)
  {
    starpu_conf_init(&conf_);
    conf_.sched_policy_name = sched_policy;
    if (starpu_init(&conf_) != 0)
    {		
      throw std::runtime_error("StarPU initialization error");
    }

    starpu_codelet_init(&codelet_);
    codelet_.cpu_funcs[0] = cpu_codelet_func;
    codelet_.cpu_funcs_name[0] = "cpu_codelet_func";
    codelet_.modes[0] = STARPU_RW;
    codelet_.name = "cpu_double";
  }

  ~StarPUSetup()
  {
    starpu_shutdown();
  }

  struct starpu_codelet* codelet() { return &codelet_; }

 private:
  static void cpu_codelet_func(void *buffers[], void *cl_arg)
  {
    std::cout << "Hello World" << std::endl;
  }

  struct starpu_conf conf_;
  struct starpu_codelet codelet_;
};

#endif // STARPU_SETUP_HPP