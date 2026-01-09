--Check duplicates
WITH MinuteBuckets AS (
    SELECT 
        Symbol,
        DATEADD(MINUTE, DATEDIFF(MINUTE, 0, PriceTimestamp), 0) AS MinuteTimestamp
    FROM tblRawPrices_Staging
    WHERE Exported = 1
),
Dupes AS (
    SELECT 
        Symbol,
        MinuteTimestamp
    FROM MinuteBuckets
    GROUP BY Symbol, MinuteTimestamp
    HAVING COUNT(*) > 1
)
SELECT s.*
FROM tblRawPrices_Staging s
JOIN Dupes d
    ON d.Symbol = s.Symbol
   AND d.MinuteTimestamp = DATEADD(MINUTE, DATEDIFF(MINUTE, 0, s.PriceTimestamp), 0)
ORDER BY s.Symbol, s.PriceTimestamp;


WITH MinuteBuckets AS (
    SELECT 
        Symbol,
        DATEADD(MINUTE, DATEDIFF(MINUTE, 0, PriceTimestamp), 0) AS MinuteTimestamp,
        COUNT(*) AS Cnt
    FROM tblRawPrices_Staging
    GROUP BY 
        Symbol,
        DATEADD(MINUTE, DATEDIFF(MINUTE, 0, PriceTimestamp), 0)
)
SELECT *
FROM MinuteBuckets
WHERE Cnt > 1
ORDER BY Symbol, MinuteTimestamp;


WITH MinuteBuckets AS (
    SELECT 
        Symbol,
        DATEADD(MINUTE, DATEDIFF(MINUTE, 0, PriceTimestamp), 0) AS MinuteTimestamp
    FROM tblRawPrices_Staging
),
Dupes AS (
    SELECT 
        Symbol,
        MinuteTimestamp
    FROM MinuteBuckets
    GROUP BY Symbol, MinuteTimestamp
    HAVING COUNT(*) > 1
)
SELECT s.*
FROM tblRawPrices_Staging s
JOIN Dupes d
    ON d.Symbol = s.Symbol
   AND d.MinuteTimestamp = DATEADD(MINUTE, DATEDIFF(MINUTE, 0, s.PriceTimestamp), 0)
ORDER BY s.Symbol, s.PriceTimestamp;


select * from tblIndicators
SELECT 
    COUNT(*) AS RowsThisRun
FROM dbo.tblIndicators
WHERE addDateTime >= '2026-01-03T22:40:00'
  AND addDateTime <= '2026-01-03T23:00:00';


select * from [dbo].[tblMergedFeatures]

EXEC sp_help 'dbo.tblMergedFeatures';
