CREATE TABLE dbo.tblSignalFeatures (
    Id BIGINT IDENTITY(1,1) PRIMARY KEY,

    -- Core identifiers
    Symbol VARCHAR(20) NOT NULL,
    FeatureTimestamp DATETIME NOT NULL,

    -- Price features
    OpenPrice FLOAT NULL,
    HighPrice FLOAT NULL,
    LowPrice FLOAT NULL,
    ClosePrice FLOAT NULL,
    Volume BIGINT NULL,

    -- Technical indicators
    ADX_14 FLOAT NULL,
    ATR_14 FLOAT NULL,
    BBANDS_LOWER_20 FLOAT NULL,
    BBANDS_MIDDLE_20 FLOAT NULL,
    BBANDS_UPPER_20 FLOAT NULL,
    BOLL_PCTB_20 FLOAT NULL,
    BOLL_WIDTH_20 FLOAT NULL,
    CCI_20 FLOAT NULL,
    EMA_9 FLOAT NULL,
    EMA_20 FLOAT NULL,
    EMA_50 FLOAT NULL,
    EMA_200 FLOAT NULL,
    RSI_14 FLOAT NULL,
    SMA_20 FLOAT NULL,
    SMA_200 FLOAT NULL,
    STOCH_K_14 FLOAT NULL,
    STOCH_D_14 FLOAT NULL,
    SUPERTREND_10 FLOAT NULL,
    WILLIAMS_R_14 FLOAT NULL,

    -- Contextual features
    DayOfWeek INT NULL,
    IsRegularSession BIT NULL,
    IsAfterHours BIT NULL,

    -- Derived ML features
    Return_1 FLOAT NULL,
    Return_5 FLOAT NULL,
    Return_20 FLOAT NULL,
    Volatility_20 FLOAT NULL,
    Momentum_10 FLOAT NULL,
    Range_Pct FLOAT NULL,
    Body_Pct FLOAT NULL,

    -- Target
    Target FLOAT NULL,

    -- Metadata
    SignalVersion INT NOT NULL,
    FeatureRunId VARCHAR(100) NOT NULL,
    RunDateTime DATETIME NOT NULL,
    MissingFeatureCount INT NULL,
    SourceIndicatorCount INT NULL
);