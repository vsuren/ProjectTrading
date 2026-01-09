-- Row counts today vs yesterday for all pipeline tables
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
SELECT TableName,
       TodayCount,
       YesterdayCount,
       TodayCount - YesterdayCount AS Increment
FROM counts;