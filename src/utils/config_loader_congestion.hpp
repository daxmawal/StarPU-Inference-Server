auto
parse_non_negative_ms(const YAML::Node& congestion_node, std::string_view key)
    -> double
{
  const auto value =
      parse_scalar<double>(congestion_node[key], key, "a number");
  if (value < 0.0) {
    throw std::invalid_argument(std::format("{} must be >= 0", key));
  }
  return value;
}

auto
parse_positive_int(const YAML::Node& congestion_node, std::string_view key)
    -> int
{
  const auto value =
      parse_scalar<long long>(congestion_node[key], key, "an integer");
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        std::format("{} must be > 0 and fit in int", key));
  }
  return static_cast<int>(value);
}

void
parse_congestion_enabled(const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["enabled"]) {
    cfg.congestion.enabled = parse_scalar<bool>(
        congestion_node["enabled"], "congestion.enabled", "a boolean");
  }
}

void
parse_congestion_latency(const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["latency_slo_ms"]) {
    cfg.congestion.latency_slo_ms =
        parse_non_negative_ms(congestion_node, "latency_slo_ms");
  }
  if (congestion_node["queue_latency_budget_ms"]) {
    cfg.congestion.queue_latency_budget_ms =
        parse_non_negative_ms(congestion_node, "queue_latency_budget_ms");
  }
  if (congestion_node["queue_latency_budget_ratio"]) {
    const auto ratio = parse_scalar<double>(
        congestion_node["queue_latency_budget_ratio"],
        "queue_latency_budget_ratio", "a number");
    if (ratio < 0.0) {
      throw std::invalid_argument("queue_latency_budget_ratio must be >= 0");
    }
    cfg.congestion.queue_latency_budget_ratio = ratio;
  }
}

void
parse_congestion_e2e_ratios(
    const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["e2e_warn_ratio"]) {
    cfg.congestion.e2e_warn_ratio = parse_scalar<double>(
        congestion_node["e2e_warn_ratio"], "e2e_warn_ratio", "a number");
  }
  if (congestion_node["e2e_ok_ratio"]) {
    cfg.congestion.e2e_ok_ratio = parse_scalar<double>(
        congestion_node["e2e_ok_ratio"], "e2e_ok_ratio", "a number");
  }
  if (cfg.congestion.e2e_warn_ratio <= 0.0) {
    throw std::invalid_argument("e2e_warn_ratio must be > 0");
  }
  if (cfg.congestion.e2e_ok_ratio <= 0.0 ||
      cfg.congestion.e2e_ok_ratio > cfg.congestion.e2e_warn_ratio) {
    throw std::invalid_argument(
        "e2e_ok_ratio must be > 0 and <= e2e_warn_ratio");
  }
}

void
parse_congestion_fill(const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["fill_high"]) {
    cfg.congestion.fill_high = parse_scalar<double>(
        congestion_node["fill_high"], "fill_high", "a number");
  }
  if (congestion_node["fill_low"]) {
    cfg.congestion.fill_low = parse_scalar<double>(
        congestion_node["fill_low"], "fill_low", "a number");
  }
  if (cfg.congestion.fill_high <= 0.0 || cfg.congestion.fill_high > 1.0 ||
      cfg.congestion.fill_low < 0.0 || cfg.congestion.fill_low >= 1.0) {
    throw std::invalid_argument(
        "fill_high must be (0,1] and fill_low in [0,1)");
  }
  if (cfg.congestion.fill_low >= cfg.congestion.fill_high) {
    throw std::invalid_argument("fill_low must be < fill_high");
  }
}

void
parse_congestion_rho(const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["rho_high"]) {
    cfg.congestion.rho_high = parse_scalar<double>(
        congestion_node["rho_high"], "rho_high", "a number");
  }
  if (congestion_node["rho_low"]) {
    cfg.congestion.rho_low =
        parse_scalar<double>(congestion_node["rho_low"], "rho_low", "a number");
  }
  if (cfg.congestion.rho_high <= 0.0) {
    throw std::invalid_argument("rho_high must be > 0");
  }
  if (cfg.congestion.rho_low < 0.0) {
    throw std::invalid_argument("rho_low must be >= 0");
  }
  if (cfg.congestion.rho_low >= cfg.congestion.rho_high) {
    throw std::invalid_argument("rho_low must be < rho_high");
  }
}

void
parse_congestion_ewma(const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["alpha_ewma"]) {
    cfg.congestion.alpha = parse_scalar<double>(
        congestion_node["alpha_ewma"], "alpha_ewma", "a number");
  }
  if (cfg.congestion.alpha <= 0.0 || cfg.congestion.alpha > 1.0) {
    throw std::invalid_argument("alpha_ewma must be in (0, 1]");
  }
}

void
parse_congestion_horizons(const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["entry_horizon_ms"]) {
    cfg.congestion.entry_horizon_ms =
        parse_positive_int(congestion_node, "entry_horizon_ms");
  }
  if (congestion_node["exit_horizon_ms"]) {
    cfg.congestion.exit_horizon_ms =
        parse_positive_int(congestion_node, "exit_horizon_ms");
  }
  if (cfg.congestion.entry_horizon_ms <= 0 ||
      cfg.congestion.exit_horizon_ms <= 0) {
    throw std::invalid_argument(
        "entry_horizon_ms and exit_horizon_ms must be > 0");
  }
}

void
parse_congestion_tick(const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  if (congestion_node["tick_interval_ms"]) {
    cfg.congestion.tick_interval_ms = parse_scalar<int>(
        congestion_node["tick_interval_ms"], "tick_interval_ms", "an integer");
  }
  if (cfg.congestion.tick_interval_ms <= 0) {
    throw std::invalid_argument("tick_interval_ms must be > 0");
  }
}

void
parse_congestion(const YAML::Node& root, RuntimeConfig& cfg)
{
  const YAML::Node congestion_node = root["congestion"];
  if (!congestion_node) {
    return;
  }
  if (!congestion_node.IsMap()) {
    throw std::invalid_argument("congestion must be a mapping");
  }
  parse_congestion_enabled(congestion_node, cfg);
  parse_congestion_latency(congestion_node, cfg);
  parse_congestion_e2e_ratios(congestion_node, cfg);
  parse_congestion_fill(congestion_node, cfg);
  parse_congestion_rho(congestion_node, cfg);
  parse_congestion_ewma(congestion_node, cfg);
  parse_congestion_horizons(congestion_node, cfg);
  parse_congestion_tick(congestion_node, cfg);
}
