# Dakshina Kannada Synthetic Solar Dataset Schema

| Column | Type | Description |
| --- | --- | --- |
| datetime | string (ISO 8601) | Timestamp for each hourly record (local time). |
| hour | int | Hour of day (0-23). |
| day | int | Day of month (1-31). |
| month | int | Month of year (1-12). |
| day_of_year | int | Day of year (1-365). |
| season | string | winter, pre_monsoon, monsoon, post_monsoon. |
| is_monsoon | int | 1 if monsoon season, else 0. |
| temperature_c | float | Ambient temperature (°C). |
| humidity_pct | float | Relative humidity (%). |
| wind_speed_mps | float | Wind speed (m/s). |
| cloud_cover_pct | float | Cloud cover (%). |
| rainfall_mm | float | Hourly rainfall (mm). |
| pressure_hpa | float | Atmospheric pressure (hPa). |
| solar_radiation_wm2 | float | Target solar radiation (W/m²). |
