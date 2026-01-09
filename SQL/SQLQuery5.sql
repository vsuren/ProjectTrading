/* ============================================================
   1. DAILY SIGNAL ROW COUNTS (BY MARKET DATE)
   ============================================================ */
SELECT 
    CAST(FeatureTimestamp AS DATE) AS TradeDate,
    COUNT(*) AS SignalRowCount
FROM dbo.tblSignalFeatures
GROUP BY CAST(FeatureTimestamp AS DATE)
ORDER BY TradeDate DESC;


/* ============================================================
   2. PER-SYMBOL SIGNAL ROW COUNTS PER DAY
   ============================================================ */
SELECT 
    CAST(FeatureTimestamp AS DATE) AS TradeDate,
    Symbol,
    COUNT(*) AS SignalRowCount
FROM dbo.tblSignalFeatures
GROUP BY CAST(FeatureTimestamp AS DATE), Symbol
ORDER BY TradeDate DESC, Symbol;


/* ============================================================
   3. MISSING BAR DETECTION (FLAG DAYS WITH LOW COVERAGE)
   ============================================================ */
SELECT 
    CAST(FeatureTimestamp AS DATE) AS TradeDate,
    Symbol,
    COUNT(*) AS SignalRowCount
FROM dbo.tblSignalFeatures
GROUP BY CAST(FeatureTimestamp AS DATE), Symbol
HAVING COUNT(*) < 300   -- adjust threshold if needed
ORDER BY TradeDate DESC, Symbol;


/* ============================================================
   4. ROWS INSERTED TODAY (BY SYSTEM TIME)
   ============================================================ */
SELECT 
    COUNT(*) AS RowsInsertedToday
FROM dbo.tblSignalFeatures
WHERE RunDateTime >= CAST(GETDATE() AS DATE);


/* ============================================================
   5. ROWS INSERTED TODAY (BY RUN ID PREFIX)
   ============================================================ */
SELECT 
    COUNT(*) AS RowsInsertedTodayByRunId
FROM dbo.tblSignalFeatures
WHERE FeatureRunId LIKE 'SIGNAL_' + CONVERT(VARCHAR(8), GETDATE(), 112) + '%';


/* ============================================================
   6. PIPELINE TABLE COUNTS (TODAY VS YESTERDAY)
   ============================================================ */
WITH counts AS (
    SELECT 'tblRawPrices_Staging' AS TableName,
           COUNT(*) AS TodayCount,
           (SELECT COUNT(*) FROM tblRawPrices_Staging 
            WHERE PriceTimestamp < CAST(GETDATE() AS DATE)) AS YesterdayCount
    FROM tblRawPrices_Staging

    UNION ALL
    SELECT 'tblRawPrices',
           COUNT(*),
           (SELECT COUNT(*) FROM tblRawPrices 
            WHERE PriceTimestamp < CAST(GETDATE() AS DATE))
    FROM tblRawPrices

    UNION ALL
    SELECT 'tblIndicators',
           COUNT(*),
           (SELECT COUNT(*) FROM tblIndicators 
            WHERE PriceTimestamp < CAST(GETDATE() AS DATE))
    FROM tblIndicators

    UNION ALL
    SELECT 'tblMergedFeatures',
           COUNT(*),
           (SELECT COUNT(*) FROM tblMergedFeatures 
            WHERE PriceTimestamp < CAST(GETDATE() AS DATE))
    FROM tblMergedFeatures

    UNION ALL
    SELECT 'tblSignalFeatures',
           COUNT(*),
           (SELECT COUNT(*) FROM tblSignalFeatures 
            WHERE FeatureTimestamp < CAST(GETDATE() AS DATE))
    FROM tblSignalFeatures
)
SELECT 
    TableName,
    TodayCount,
    YesterdayCount,
    TodayCount - YesterdayCount AS Increment
FROM counts
ORDER BY TableName;


/* ============================================================
   7. DUPLICATE DETECTION (ALL PIPELINE TABLES)
   ============================================================ */

-- RawPrices_Staging
SELECT 'tblRawPrices_Staging' AS TableName, Symbol, PriceTimestamp, COUNT(*) AS Cnt
FROM tblRawPrices_Staging
GROUP BY Symbol, PriceTimestamp
HAVING COUNT(*) > 1;

-- RawPrices
SELECT 'tblRawPrices' AS TableName, Symbol, PriceTimestamp, COUNT(*) AS Cnt
FROM tblRawPrices
GROUP BY Symbol, PriceTimestamp
HAVING COUNT(*) > 1;

-- Indicators
SELECT 'tblIndicators' AS TableName, Symbol, PriceTimestamp, IndicatorId, COUNT(*) AS Cnt
FROM tblIndicators
GROUP BY Symbol, PriceTimestamp, IndicatorId
HAVING COUNT(*) > 1;

-- MergedFeatures
SELECT 'tblMergedFeatures' AS TableName, Symbol, PriceTimestamp, COUNT(*) AS Cnt
FROM tblMergedFeatures
GROUP BY Symbol, PriceTimestamp
HAVING COUNT(*) > 1;

-- SignalFeatures
SELECT 'tblSignalFeatures' AS TableName, Symbol, FeatureTimestamp, COUNT(*) AS Cnt
FROM tblSignalFeatures
GROUP BY Symbol, FeatureTimestamp
HAVING COUNT(*) > 1;