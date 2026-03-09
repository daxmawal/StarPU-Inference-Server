// CI/build safety guard for the production server target.
// This file is compiled only in the `starpu_server` target.
#if defined(STARPU_TESTING)
#error "starpu_server must not be compiled with STARPU_TESTING"
#endif
