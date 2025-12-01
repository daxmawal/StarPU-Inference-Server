#include "signal_handler.hpp"

namespace starpu_server {

auto
server_context() -> ServerContext&
{
  static ServerContext ctx;
  return ctx;
}

void
signal_handler(int /*signal*/)
{
  server_context().stop_requested.store(true);
}

}  // namespace starpu_server
