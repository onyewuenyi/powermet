# Run directory template

One directory per design × build, produced by the flows (or copied/linked from where they write):

```
<root>/<design>/<build>/
  metadata.json                        <- templates/metadata.template.json
  mapping/fub_map.csv                  <- templates/fub_map.template.csv (export of the model root)
  pprtl/<wl>_<op>/{rtl,physical}_power.rpt
  primepower/<wl>_<op>/power_hier.rpt
  primetime/<op>/timing_summary.rpt
  starrc/parasitics_summary.rpt
  implementation/qor_summary.rpt
  activity/<wl>.saif                   <- output of the FSDB -> SAIF flow
  perf/<wl>_<op>.csv
```

If the target environment lays files out differently, do not move the files: set `source_patterns`
in `.powermet/config.toml` (see `powermet sources`) or edit `get_files()` in the adapter.
