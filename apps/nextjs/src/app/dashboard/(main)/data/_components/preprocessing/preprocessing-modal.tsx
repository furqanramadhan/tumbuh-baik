import React, { useState, useEffect, useRef, useCallback } from "react";
import { Icons } from "@/app/dashboard/_components/icons";
import {
  preprocessNasaDatasetWithStream,
  preprocessBmkgDatasetWithStream,
} from "@/lib/fetch/files.fetch";
import { useRouter } from "next/navigation";
import { ScrollArea } from "@/components/ui/scroll-area";

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface LogEntry {
  type: "log" | "progress" | "error" | "complete" | "info" | "success";
  level?: string;
  message: string;
  timestamp: number;
}

/**
 * Unified preprocessing result interface that handles both NASA and BMKG datasets
 */
interface PreprocessingResult {
  // Basic counts
  recordCount?: number;
  originalRecordCount?: number;
  cleanedCollection?: string;
  collection?: string;
  message?: string;
  preprocessedCollections?: string[];
  
  // Comprehensive preprocessing report
  preprocessing_report?: {
    // Dataset type
    dataset_type?: "nasa" | "bmkg";
    
    // Missing data handling (both datasets)
    missing_data?: {
      fill_values_replaced?: Record<string, number>;  // NASA: -999, BMKG: 8888
      tail_data_excluded?: {                          // NASA only
        count: number;
        latest_complete_date: string;
        reason: string;
      };
      imputed_values?: Record<string, number>;        // Both datasets
      suspicious_zeros_fixed?: number;                // BMKG only (FF_AVG)
    };
    
    // Outliers (both datasets)
    outliers?: {
      total_outliers: number;
      by_parameter: Record<string, number>;
      methods_used?: string[];                        // NASA only
      treatment?: string;                             // NASA only
    };
    
    // Smoothing (NASA only - BMKG doesn't use smoothing)
    smoothing?: {
      method?: string;
      parameters_smoothed?: Record<string, string>;
      decisions?: {
        smoothed: string[];
        skipped: string[];
        reasons: Record<string, string>;
      };
      summary?: {
        total_parameters: number;
        smoothed_count: number;
        skipped_count: number;
      };
    };
    
    // Smoothing validation (NASA only)
    smoothing_validation?: Record<string, {
      gcv_score?: number;
      trend_preservation_pct?: number;
      quality_status?: "excellent" | "good" | "fair" | "poor";
      smoothing_method?: string;
      data_points?: number;
    }>;
    
    // Alpha optimization (NASA only - optional)
    alpha_optimization?: Record<string, {
      original_alpha: number;
      optimized_alpha: number;
      improvement: string;
    }>;
    
    // Gaps detection (both datasets)
    gaps?: {
      total_gaps: number;
      small_gaps?: number;
      medium_gaps?: number;
      large_gaps?: number;
      gap_details?: Array<{
        start_date: string;
        end_date: string;
        duration_days: number;
        type: string;
        imputation_method: string;
      }>;
    };
    
    // Model coverage (both datasets)
    model_coverage?: {
      holt_winters: {
        coverage_percentage: number;
        uncovered_breakdown: Record<string, number>;
        model_suitability?: string;
      };
      lstm: {
        coverage_percentage: number;
        uncovered_breakdown: Record<string, number>;
        model_suitability?: string;
      };
      per_parameter?: Record<string, any>;
    };
    
    // Quality metrics (both datasets)
    quality_metrics?: {
      original_records: number;
      processed_records: number;
      records_removed: number;
      completeness_percentage: number;
      data_quality: "high" | "medium" | "low";
    };
    
    // STL Decomposition (both datasets)
    decomposition?: {
      parameters_decomposed: string[];
      decomposition_data: Record<string, any>;
    };
    
    // Warnings (both datasets)
    warnings?: string[];
  };
}

interface PreprocessingModalProps {
  collectionName: string;
  isAPI: boolean;  // true = NASA POWER, false = BMKG
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (result: PreprocessingResult) => void;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const MAX_LOG_ENTRIES = 1000;

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PreprocessingModal({
  collectionName,
  isAPI,
  isOpen,
  onClose,
  onSuccess,
}: PreprocessingModalProps) {
  // ---------------------------------------------------------------------------
  // STATE
  // ---------------------------------------------------------------------------
  
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [progress, setProgress] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState("");
  const [status, setStatus] = useState<"processing" | "success" | "error" | "idle">("idle");
  const [result, setResult] = useState<PreprocessingResult | null>(null);
  
  // ---------------------------------------------------------------------------
  // REFS
  // ---------------------------------------------------------------------------
  
  const eventSourceRef = useRef<EventSource | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  // ---------------------------------------------------------------------------
  // DATASET CONFIGURATION
  // ---------------------------------------------------------------------------
  
  const datasetConfig = {
    displayName: isAPI ? 'NASA POWER' : 'BMKG',
    streamFunction: isAPI ? preprocessNasaDatasetWithStream : preprocessBmkgDatasetWithStream,
    color: isAPI ? 'blue' : 'green',
  };

  // ---------------------------------------------------------------------------
  // UTILITY FUNCTIONS
  // ---------------------------------------------------------------------------
  
  const cleanup = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    if (status !== "processing") return;
    
    requestAnimationFrame(() => {
      const viewport = scrollAreaRef.current?.querySelector("[data-radix-scroll-area-viewport]");
      if (viewport) {
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: "smooth" });
      }
    });
  }, [status]);

  const addLog = useCallback((type: LogEntry["type"], message: string, level = "INFO") => {
    setLogs(prev => {
      const newLog = { type, message, level, timestamp: Date.now() };
      const updated = [...prev, newLog];
      return updated.length > MAX_LOG_ENTRIES ? updated.slice(-MAX_LOG_ENTRIES) : updated;
    });
  }, []);

  // ---------------------------------------------------------------------------
  // EFFECTS
  // ---------------------------------------------------------------------------
  
  // Lock body scroll when modal is open
  useEffect(() => {
    if (!isOpen) return;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; };
  }, [isOpen]);

  // Auto-scroll logs when processing
  useEffect(() => {
    if (status === "processing" && logs.length > 0) scrollToBottom();
  }, [logs.length, status, scrollToBottom]);

  // Start preprocessing when modal opens
  useEffect(() => {
    if (isOpen && status === "idle") startPreprocessing();
    return cleanup;
  }, [isOpen, cleanup]);

  // ---------------------------------------------------------------------------
  // PREPROCESSING LOGIC
  // ---------------------------------------------------------------------------
  
  const startPreprocessing = () => {
    setStatus("processing");
    setLogs([]);
    setProgress(0);
    setCurrentStage("Connecting...");
    setResult(null);

    const callbacks = {
      onLog: (logData: any) => addLog("log", logData.message, logData.level || "INFO"),
      
      onProgress: (progressPercent: number, stage: string, message: string) => {
        const safeProgress = Math.max(0, Math.min(100, progressPercent || 0));
        setProgress(safeProgress);
        setCurrentStage(message || stage || "Processing...");
        addLog("progress", `[${safeProgress}%] ${message}`);
      },
      
      onComplete: (completionResult: any) => {
        setStatus("success");
        setResult(completionResult);
        addLog("success", `✅ ${datasetConfig.displayName} preprocessing completed!`, "SUCCESS");
      },
      
      onError: (errorMessage: string) => {
        setStatus("error");
        addLog("error", `❌ Error: ${errorMessage || "Unknown error"}`, "ERROR");
      }
    };

    try {
      addLog("info", `Starting ${datasetConfig.displayName} preprocessing...`);
      const streamResult = datasetConfig.streamFunction(
        collectionName,
        callbacks.onLog,
        callbacks.onProgress,
        callbacks.onComplete,
        callbacks.onError
      );
      eventSourceRef.current = streamResult.eventSource;
      addLog("info", `Connected to ${datasetConfig.displayName} preprocessing server...`);
    } catch (error) {
      setStatus("error");
      addLog("error", `❌ Failed to start: ${error instanceof Error ? error.message : "Unknown error"}`, "ERROR");
    }
  };

  // ---------------------------------------------------------------------------
  // EVENT HANDLERS
  // ---------------------------------------------------------------------------
  
  const handleClose = useCallback(() => {
    if (status === "processing" && !window.confirm("Stop processing and close?")) return;
    
    cleanup();
    setStatus("idle");
    setLogs([]);
    setProgress(0);
    setCurrentStage("");
    setResult(null);
    onClose();
  }, [status, cleanup, onClose]);

  const handleSuccess = useCallback(() => {
    if (status === "success" && result?.cleanedCollection) {
      cleanup();
      if (onSuccess) onSuccess(result);
      router.push(`/dashboard/data/${encodeURIComponent(result.cleanedCollection)}`);
      onClose();
    }
  }, [status, result, onSuccess, router, cleanup, onClose]);

  // ---------------------------------------------------------------------------
  // UI HELPERS
  // ---------------------------------------------------------------------------
  
  const getStatusIcon = () => {
    const iconMap = {
      processing: <Icons.spinner className="h-6 w-6 animate-spin text-blue-500" />,
      success: <Icons.checked className="h-6 w-6 text-green-500" />,
      error: <Icons.closeX className="h-6 w-6 text-red-500" />,
      idle: <Icons.alertCircle className="h-6 w-6 text-gray-400" />
    };
    return iconMap[status];
  };

  /**
   * Get comprehensive result metrics that work for both NASA and BMKG datasets
   * Handles all preprocessing report fields dynamically
   */
  const getResultMetrics = () => {
    if (!result) return [];

    const report = result.preprocessing_report;
    const qualityMetrics = report?.quality_metrics;
    const missingData = report?.missing_data;
    const smoothing = report?.smoothing;
    const gaps = report?.gaps;
    const coverage = report?.model_coverage;
    
    const metrics: Array<{
      label: string;
      value: string | number;
      isBreakable?: boolean;
      category?: string;
    }> = [];

    // ========================================================================
    // BASIC COUNTS
    // ========================================================================
    
    if (qualityMetrics?.original_records) {
      metrics.push({
        label: "Original Records",
        value: qualityMetrics.original_records.toLocaleString(),
        category: "basic"
      });
    }
    
    if (qualityMetrics?.processed_records) {
      metrics.push({
        label: "Processed Records",
        value: qualityMetrics.processed_records.toLocaleString(),
        category: "basic"
      });
    }
    
    if (qualityMetrics?.records_removed) {
      metrics.push({
        label: "Records Removed",
        value: qualityMetrics.records_removed.toLocaleString(),
        category: "basic"
      });
    }

    // ========================================================================
    // MISSING DATA HANDLING
    // ========================================================================
    
    // Fill values replaced (both datasets)
    if (missingData?.fill_values_replaced) {
      const totalFillValues = Object.values(missingData.fill_values_replaced).reduce(
        (sum, count) => sum + count, 0
      );
      if (totalFillValues > 0) {
        metrics.push({
          label: isAPI ? "Fill Values Replaced (-999)" : "Fill Values Replaced (8888)",
          value: totalFillValues.toLocaleString(),
          category: "imputation"
        });
      }
    }
    
    // Suspicious zeros fixed (BMKG only)
    if (!isAPI && missingData?.suspicious_zeros_fixed) {
      metrics.push({
        label: "Suspicious FF_AVG Zeros Fixed",
        value: missingData.suspicious_zeros_fixed.toLocaleString(),
        category: "imputation"
      });
    }
    
    // Tail data excluded (NASA only)
    if (isAPI && missingData?.tail_data_excluded) {
      metrics.push({
        label: "Tail Data Excluded (NASA lag)",
        value: missingData.tail_data_excluded.count.toLocaleString(),
        category: "imputation"
      });
    }
    
    // Total imputed values
    if (missingData?.imputed_values) {
      const totalImputed = Object.values(missingData.imputed_values).reduce(
        (sum, count) => sum + count, 0
      );
      if (totalImputed > 0) {
        metrics.push({
          label: "Total Values Imputed",
          value: totalImputed.toLocaleString(),
          category: "imputation"
        });
      }
    }

    // ========================================================================
    // OUTLIERS
    // ========================================================================
    
    if (report?.outliers?.total_outliers !== undefined) {
      metrics.push({
        label: "Outliers Detected & Handled",
        value: report.outliers.total_outliers.toLocaleString(),
        category: "outliers"
      });
    }

    // ========================================================================
    // GAPS
    // ========================================================================
    
    if (gaps?.total_gaps !== undefined) {
      metrics.push({
        label: "Time Series Gaps Detected",
        value: gaps.total_gaps.toLocaleString(),
        category: "gaps"
      });
    }
    
    if (gaps?.large_gaps) {
      metrics.push({
        label: "Large Gaps (>90 days)",
        value: gaps.large_gaps.toLocaleString(),
        category: "gaps"
      });
    }

    // ========================================================================
    // SMOOTHING (NASA only)
    // ========================================================================
    
    if (isAPI && smoothing?.summary) {
      metrics.push({
        label: "Parameters Smoothed",
        value: `${smoothing.summary.smoothed_count}/${smoothing.summary.total_parameters}`,
        category: "smoothing"
      });
    }
    
    // Average smoothing quality (NASA only)
    if (isAPI && report?.smoothing_validation) {
      const validationEntries = Object.values(report.smoothing_validation);
      const avgTrend = validationEntries.reduce((sum, v: any) => 
        sum + (v.trend_preservation_pct || 0), 0
      ) / validationEntries.length;
      
      if (avgTrend > 0) {
        metrics.push({
          label: "Avg Trend Preservation",
          value: `${avgTrend.toFixed(1)}%`,
          category: "smoothing"
        });
      }
    }

    // ========================================================================
    // MODEL COVERAGE
    // ========================================================================
    
    if (coverage?.holt_winters?.coverage_percentage !== undefined) {
      metrics.push({
        label: "Holt-Winters Coverage",
        value: `${coverage.holt_winters.coverage_percentage.toFixed(1)}%`,
        category: "coverage"
      });
    }
    
    if (coverage?.lstm?.coverage_percentage !== undefined) {
      metrics.push({
        label: "LSTM Coverage",
        value: `${coverage.lstm.coverage_percentage.toFixed(1)}%`,
        category: "coverage"
      });
    }

    // ========================================================================
    // QUALITY METRICS
    // ========================================================================
    
    if (qualityMetrics?.completeness_percentage !== undefined) {
      metrics.push({
        label: "Data Completeness",
        value: `${qualityMetrics.completeness_percentage.toFixed(1)}%`,
        category: "quality"
      });
    }
    
    if (qualityMetrics?.data_quality) {
      metrics.push({
        label: "Overall Quality",
        value: qualityMetrics.data_quality.toUpperCase(),
        category: "quality"
      });
    }

    // ========================================================================
    // DECOMPOSITION
    // ========================================================================
    
    if (report?.decomposition?.parameters_decomposed) {
      metrics.push({
        label: "STL Decomposition Applied",
        value: `${report.decomposition.parameters_decomposed.length} parameters`,
        category: "decomposition"
      });
    }

    // ========================================================================
    // COLLECTION INFO
    // ========================================================================
    
    if (result.cleanedCollection) {
      metrics.push({
        label: "Cleaned Collection",
        value: result.cleanedCollection,
        isBreakable: true,
        category: "output"
      });
    }

    return metrics;
  };

  // ---------------------------------------------------------------------------
  // RENDER
  // ---------------------------------------------------------------------------
  
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      role="dialog"
      aria-modal="true"
      onClick={handleClose}
    >
      <div
        className="bg-white rounded-lg shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ================================================================== */}
        {/* HEADER */}
        {/* ================================================================== */}
        
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Preprocessing {datasetConfig.displayName} Data
              </h2>
              <p className="text-sm text-gray-500 mt-1">{collectionName}</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            disabled={status === "processing"}
          >
            <Icons.closeX className="h-6 w-6" />
          </button>
        </div>

        {/* ================================================================== */}
        {/* PROGRESS BAR */}
        {/* ================================================================== */}
        
        {status === "processing" && (
          <div className="px-6 pt-4">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-600">{currentStage}</span>
              <span className="font-medium text-blue-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* LOGS */}
        {/* ================================================================== */}
        
        <div className="p-6 flex flex-col flex-1 min-h-0">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm text-gray-600 font-medium">
              Live Logs ({logs.length})
            </span>
            {status === "processing" && (
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-gray-500">Auto-scrolling</span>
              </div>
            )}
          </div>

          <ScrollArea ref={scrollAreaRef} className="h-[400px] border rounded-lg">
            <div className="bg-gray-900 p-4">
              {logs.length === 0 ? (
                <div className="text-gray-400 text-center py-8 font-mono text-sm">
                  Waiting for logs...
                </div>
              ) : (
                <div className="space-y-1 font-mono text-sm">
                  {logs.map((log, index) => (
                    <div
                      key={`${log.timestamp}-${index}`}
                      className={
                        log.level === "ERROR" ? "text-red-400" :
                        log.level === "WARNING" ? "text-yellow-400" :
                        log.level === "SUCCESS" ? "text-green-400" :
                        "text-gray-300"
                      }
                    >
                      <span className="text-gray-500 mr-2">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      {log.message}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* ================================================================== */}
        {/* COMPREHENSIVE RESULTS SUMMARY */}
        {/* ================================================================== */}
        
        {status === "success" && result && (
          <div className="px-6 pb-4 max-h-[300px] overflow-y-auto">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h3 className="font-semibold text-green-900 mb-3 flex items-center gap-2">
                <Icons.checked className="h-5 w-5" />
                Processing Complete - Comprehensive Summary
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2 text-sm">
                {getResultMetrics().map((metric, index) => (
                  <div 
                    key={index} 
                    className={`flex justify-between items-start py-1 ${
                      metric.category === 'output' ? 'col-span-2' : ''
                    }`}
                  >
                    <span className="text-gray-700 font-medium mr-2">
                      {metric.label}:
                    </span>
                    <span className={`font-semibold text-right ${
                      metric.isBreakable 
                        ? 'text-xs break-all max-w-[250px]' 
                        : ''
                    } ${
                      metric.category === 'quality' 
                        ? 'text-green-700' 
                        : metric.category === 'coverage'
                        ? 'text-blue-700'
                        : 'text-gray-900'
                    }`}>
                      {metric.value}
                    </span>
                  </div>
                ))}
              </div>
              
              {/* Warnings section */}
              {result.preprocessing_report?.warnings && 
               result.preprocessing_report.warnings.length > 0 && (
                <div className="mt-4 pt-3 border-t border-green-300">
                  <h4 className="text-sm font-semibold text-yellow-800 mb-2">
                    ⚠️ Warnings ({result.preprocessing_report.warnings.length})
                  </h4>
                  <ul className="text-xs text-yellow-700 space-y-1 max-h-24 overflow-y-auto">
                    {result.preprocessing_report.warnings.slice(0, 5).map((warning, idx) => (
                      <li key={idx} className="flex items-start gap-1">
                        <span className="text-yellow-600 mt-0.5">•</span>
                        <span>{warning}</span>
                      </li>
                    ))}
                    {result.preprocessing_report.warnings.length > 5 && (
                      <li className="text-gray-600 italic">
                        ... and {result.preprocessing_report.warnings.length - 5} more warnings
                      </li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* FOOTER */}
        {/* ================================================================== */}
        
        <div className="flex justify-between items-center gap-3 p-6 border-t bg-gray-50">
          <div className="flex items-center gap-2 text-sm">
            {status === "success" && (
              <>
                <Icons.checked className="h-4 w-4 text-green-600" />
                <span className="text-green-600">Ready to view cleaned dataset!</span>
              </>
            )}
            {status === "processing" && (
              <>
                <Icons.spinner className="h-4 w-4 animate-spin text-gray-600" />
                <span className="text-gray-600">Processing {datasetConfig.displayName} data...</span>
              </>
            )}
            {status === "error" && (
              <>
                <Icons.closeX className="h-4 w-4 text-red-600" />
                <span className="text-red-600">Processing failed - Check logs</span>
              </>
            )}
          </div>

          <button
            onClick={status === "success" ? handleSuccess : handleClose}
            className={`px-6 py-2 rounded-lg text-white font-medium transition-colors ${
              status === "processing" ? "bg-red-600 hover:bg-red-700" :
              status === "success" ? "bg-green-600 hover:bg-green-700" :
              "bg-gray-600 hover:bg-gray-700"
            }`}
          >
            {status === "processing" ? "Stop & Close" : 
             status === "success" ? "View Cleaned Dataset →" : 
             "Close"}
          </button>
        </div>
      </div>
    </div>
  );
}