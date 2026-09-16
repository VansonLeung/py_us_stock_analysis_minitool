from .market_data_fetchers import (
    fetch_futu_symbol_history,
    fetch_symbol_price_history,
    fetch_us_equity_symbols,
    fetch_yahoo_symbol_history,
    fetch_yahoo_symbol_history_for_period,
)
from .scan_cli_controller import parse_scan_cli_arguments, run_scan_cli_controller
from .scan_models import (
    DEFAULT_METADATA_CACHE,
    DEFAULT_METADATA_SCOPE,
    DEFAULT_METADATA_TTL_DAYS,
    EnrichedRow,
    SCAN_METADATA_COLUMNS,
    ScanRow,
    StockMetadata,
    VCPResult,
)
from .scan_result_writer import (
    _load_previous_scores,
    _load_symbols_from_csv,
    build_enriched_symbol_analysis,
    build_enriched_symbol_analysis_list,
    load_previous_scan_score_map,
    load_symbol_list_from_csv_file,
    save_enriched_analysis_outputs,
    save_scan_output_files,
)
from .stock_metadata_service import (
    attach_stock_metadata_to_scan_frame,
    build_empty_stock_metadata_model,
    load_stock_metadata_by_symbol,
)
from .vcp_pattern_analyzer import (
    analyze_symbol_vcp_state,
    detect_vcp_pattern,
    find_local_extrema_indices,
    run_vcp_scan_for_symbol_list,
)

analyze_symbol = analyze_symbol_vcp_state
attach_metadata_to_scan_frame = attach_stock_metadata_to_scan_frame
detect_vcp = detect_vcp_pattern
empty_stock_metadata = build_empty_stock_metadata_model
enrich_symbol = build_enriched_symbol_analysis
fetch_history = fetch_symbol_price_history
fetch_history_futu = fetch_futu_symbol_history
fetch_history_yahoo = fetch_yahoo_symbol_history
fetch_history_yahoo_period = fetch_yahoo_symbol_history_for_period
fetch_symbols = fetch_us_equity_symbols
get_stock_metadata_map = load_stock_metadata_by_symbol
load_previous_scores = load_previous_scan_score_map
load_symbols_from_csv = load_symbol_list_from_csv_file
local_extrema = find_local_extrema_indices
main = run_scan_cli_controller
parse_args = parse_scan_cli_arguments
run_enrichment = build_enriched_symbol_analysis_list
run_scan = run_vcp_scan_for_symbol_list
save_enrichment = save_enriched_analysis_outputs
save_outputs = save_scan_output_files
