import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';

const ReadingSettingsContext = createContext();

export const READING_SETTINGS_DEFAULTS = {
    aiAutoProcess: true,
    autoOutlineSummary: true,
    autoPretranslate: false,
    pretranslateConcurrency: 5,
    overviewDefaultDepth: 'standard',
};

const clampConcurrency = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return READING_SETTINGS_DEFAULTS.pretranslateConcurrency;
    return Math.max(1, Math.min(8, Math.round(numeric)));
};

const normalizeDepth = (value) => {
    return ['brief', 'standard', 'detailed'].includes(value) ? value : READING_SETTINGS_DEFAULTS.overviewDefaultDepth;
};

export const ReadingSettingsProvider = ({ children }) => {
    const [aiAutoProcess, setAiAutoProcess] = useState(READING_SETTINGS_DEFAULTS.aiAutoProcess);
    const [autoOutlineSummary, setAutoOutlineSummary] = useState(READING_SETTINGS_DEFAULTS.autoOutlineSummary);
    const [autoPretranslate, setAutoPretranslateState] = useState(READING_SETTINGS_DEFAULTS.autoPretranslate);
    const [pretranslateConcurrency, setPretranslateConcurrencyState] = useState(READING_SETTINGS_DEFAULTS.pretranslateConcurrency);
    const [overviewDefaultDepth, setOverviewDefaultDepthState] = useState(READING_SETTINGS_DEFAULTS.overviewDefaultDepth);

    const debounceTimerRef = useRef(null);
    const pendingSettingsRef = useRef(null);

    useEffect(() => {
        try {
            const saved = localStorage.getItem('readingSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                if (settings.aiAutoProcess !== undefined) setAiAutoProcess(Boolean(settings.aiAutoProcess));
                if (settings.autoOutlineSummary !== undefined) setAutoOutlineSummary(Boolean(settings.autoOutlineSummary));
                if (settings.autoPretranslate !== undefined) setAutoPretranslateState(Boolean(settings.autoPretranslate));
                if (settings.pretranslateConcurrency !== undefined) setPretranslateConcurrencyState(clampConcurrency(settings.pretranslateConcurrency));
                if (settings.overviewDefaultDepth !== undefined) setOverviewDefaultDepthState(normalizeDepth(settings.overviewDefaultDepth));
                return;
            }

            const legacyPretranslate = localStorage.getItem('enableHoverPretranslate');
            if (legacyPretranslate !== null) {
                setAutoPretranslateState(JSON.parse(legacyPretranslate));
            }
            const legacyConcurrency = localStorage.getItem('pretranslateConcurrency');
            if (legacyConcurrency !== null) {
                setPretranslateConcurrencyState(clampConcurrency(JSON.parse(legacyConcurrency)));
            }
            const legacyOverviewDepth = localStorage.getItem('overviewDepth');
            if (legacyOverviewDepth !== null) {
                setOverviewDefaultDepthState(normalizeDepth(JSON.parse(legacyOverviewDepth)));
            }
        } catch (error) {
            console.error('加载智能阅读设置失败:', error);
        }
    }, []);

    const flushSave = useCallback(() => {
        if (pendingSettingsRef.current === null) return;
        try {
            localStorage.setItem('readingSettings', JSON.stringify(pendingSettingsRef.current));
            localStorage.setItem('enableHoverPretranslate', JSON.stringify(pendingSettingsRef.current.autoPretranslate));
            localStorage.setItem('pretranslateConcurrency', JSON.stringify(pendingSettingsRef.current.pretranslateConcurrency));
            localStorage.setItem('overviewDepth', JSON.stringify(pendingSettingsRef.current.overviewDefaultDepth));
        } catch (error) {
            console.error('保存智能阅读设置失败:', error);
        }
        pendingSettingsRef.current = null;
    }, []);

    const debouncedSave = useCallback((settings) => {
        pendingSettingsRef.current = settings;
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
        }
        debounceTimerRef.current = setTimeout(() => {
            flushSave();
            debounceTimerRef.current = null;
        }, 500);
    }, [flushSave]);

    useEffect(() => {
        debouncedSave({
            aiAutoProcess,
            autoOutlineSummary,
            autoPretranslate,
            pretranslateConcurrency,
            overviewDefaultDepth,
        });
    }, [aiAutoProcess, autoOutlineSummary, autoPretranslate, pretranslateConcurrency, overviewDefaultDepth, debouncedSave]);

    useEffect(() => {
        const handleBeforeUnload = () => flushSave();
        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
            flushSave();
        };
    }, [flushSave]);

    const setAutoPretranslate = useCallback((value) => setAutoPretranslateState(Boolean(value)), []);
    const setPretranslateConcurrency = useCallback((value) => setPretranslateConcurrencyState(clampConcurrency(value)), []);
    const setOverviewDefaultDepth = useCallback((value) => setOverviewDefaultDepthState(normalizeDepth(value)), []);

    const resetReadingSettings = useCallback(() => {
        setAiAutoProcess(READING_SETTINGS_DEFAULTS.aiAutoProcess);
        setAutoOutlineSummary(READING_SETTINGS_DEFAULTS.autoOutlineSummary);
        setAutoPretranslateState(READING_SETTINGS_DEFAULTS.autoPretranslate);
        setPretranslateConcurrencyState(READING_SETTINGS_DEFAULTS.pretranslateConcurrency);
        setOverviewDefaultDepthState(READING_SETTINGS_DEFAULTS.overviewDefaultDepth);
    }, []);

    const value = {
        aiAutoProcess,
        autoOutlineSummary,
        autoPretranslate,
        pretranslateConcurrency,
        overviewDefaultDepth,
        setAiAutoProcess,
        setAutoOutlineSummary,
        setAutoPretranslate,
        setPretranslateConcurrency,
        setOverviewDefaultDepth,
        resetReadingSettings,
        flushSave,
        READING_SETTINGS_DEFAULTS,
    };

    return (
        <ReadingSettingsContext.Provider value={value}>
            {children}
        </ReadingSettingsContext.Provider>
    );
};

export const useReadingSettings = () => {
    const context = useContext(ReadingSettingsContext);
    if (!context) {
        throw new Error('useReadingSettings 必须在 ReadingSettingsProvider 内部使用');
    }
    return context;
};

export default ReadingSettingsContext;
