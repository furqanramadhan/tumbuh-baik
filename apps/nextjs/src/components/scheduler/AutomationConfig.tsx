"use client";

import { useState, useEffect, useCallback, use } from "react";
import { Switch } from "@headlessui/react";
import {
  getAutomationConfig,
  saveAutomationConfig,
  getAvailableDatasets,
} from "@/lib/fetch/scheduler.fetch";
import { DatasetConfig } from "@/types/scheduler";
import { Save, AlertTriangle, Loader2 } from "lucide-react";
import toast from "react-hot-toast";

const DAYS = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];

export default function AutomationConfig() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form states
  const [enabled, setEnabled] = useState(false);
  const [frequency, setFrequency] = useState<"weekly" | "biweekly">("weekly");
  const [executionTime, setExecutionTime] = useState("02:00");
  const [dayOfWeek, setDayOfWeek] = useState(0);
  const [daysOfWeek, setDaysOfWeek] = useState<number[]>([0, 3]);
  const [selectedDatasets, setSelectedDatasets] = useState<{
    nasa_refresh: string[];
    nasa_preprocess: string[];
    bmkg_preprocess: string[];
  }>({ nasa_refresh: [], nasa_preprocess: [], bmkg_preprocess: [] });

  const [availableDatasets, setAvailableDatasets] = useState<{
    nasa: DatasetConfig[];
    bmkg: DatasetConfig[];
  }>({ nasa: [], bmkg: [] });
  const [nextRunsPreview, setNextRunsPreview] = useState<string[]>([]);

  // Function to calculate frontend preview
  const calculateNextRuns = useCallback(() => {
    if (!enabled) {
      setNextRunsPreview([]);
      return;
    }

    const runs: string[] = [];
    const [hour = 2, minute = 0] = executionTime.split(":").map(Number);

    let cursor = new Date();
    // Reset to execution time
    cursor.setHours(hour, minute, 0, 0);
    if (cursor.getTime() < new Date().getTime()) {
      cursor.setDate(cursor.getDate() + 1);
    }

    while (runs.length < 5) {
      if (frequency === "weekly" && cursor.getDay() !== dayOfWeek) {
        cursor.setDate(cursor.getDate() + 1);
        continue;
      }
      if (frequency === "biweekly" && !daysOfWeek.includes(cursor.getDay())) {
        cursor.setDate(cursor.getDate() + 1);
        continue;
      }
      runs.push(cursor.toISOString());
      cursor.setDate(cursor.getDate() + 1);
    }
    setNextRunsPreview(runs);
  }, [enabled, frequency, executionTime, dayOfWeek, daysOfWeek]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [config, datasets] = await Promise.all([
        getAutomationConfig(),
        getAvailableDatasets(),
      ]);

      setEnabled(config.enabled || false);
      setFrequency(config.frequency || "weekly");
      setExecutionTime(config.executionTime || "02:00");
      setDayOfWeek(config.dayOfWeek || 0);
      setDaysOfWeek(config.daysOfWeek || [0, 3]);
      setSelectedDatasets(
        config.selectedDatasets || {
          nasa_refresh: [],
          nasa_preprocess: [],
          bmkg_preprocess: [],
        },
      );
      setAvailableDatasets({
        nasa: datasets.nasa_raw,
        bmkg: datasets.bmkg_raw,
      });
    } catch (err: any) {
      setError(err.message || "Failed to load configuration");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);
  useEffect(() => {
    calculateNextRuns();
  }, [calculateNextRuns]);

  const handleSave = async () => {
    if (enabled) {
      if (frequency === "biweekly" && daysOfWeek.length !== 2) {
        return toast.error(
          "Please select exactly 2 days for bi-weekly schedule.",
        );
      }
      if (
        selectedDatasets.nasa_refresh.length === 0 &&
        selectedDatasets.nasa_preprocess.length === 0 &&
        selectedDatasets.bmkg_preprocess.length === 0
      ) {
        return toast.error("At least one dataset must be selected.");
      }
    }

    try {
      setSaving(true);
      await saveAutomationConfig({
        enabled,
        frequency,
        executionTime,
        dayOfWeek,
        daysOfWeek,
        selectedDatasets,
      });
      toast.success("Automation configuration saved successfully");
    } catch (err: any) {
      toast.error(err.message || "Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  const toggleDataset = (
    category: "nasa_refresh" | "nasa_preprocess" | "bmkg_preprocess",
    name: string,
  ) => {
    setSelectedDatasets((prev) => ({
      ...prev,
      [category]: prev[category].includes(name)
        ? prev[category].filter((d) => d !== name)
        : [...prev[category], name],
    }));
  };

  const selectAll = (
    category: "nasa_refresh" | "nasa_preprocess" | "bmkg_preprocess",
    checked: boolean,
    sourceList: DatasetConfig[],
  ) => {
    setSelectedDatasets((prev) => ({
      ...prev,
      [category]: checked ? sourceList.map((d) => d.collectionName) : [],
    }));
  };

  const toggleArrayItem = (
    setter: React.Dispatch<React.SetStateAction<number[]>>,
    val: number,
  ) => {
    setter((prev) =>
      prev.includes(val)
        ? prev.filter((x) => x !== val)
        : [...prev, val].sort((a, b) => a - b),
    );
  };

  if (loading)
    return (
      <div className="p-6 text-center text-gray-500 animate-pulse">
        Loading config...
      </div>
    );

  return (
    <div className="bg-white border rounded-xl shadow-sm overflow-hidden flex flex-col">
      <div className="p-5 border-b bg-gray-50 flex justify-between items-center">
        <div>
          <h2 className="text-lg font-semibold text-gray-800">
            Automation Settings
          </h2>
          <p className="text-sm text-gray-500">
            Configure background scheduled processing
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <span
            className={`text-sm font-medium ${enabled ? "text-blue-600" : "text-gray-400"}`}
          >
            {enabled ? "Enabled" : "Disabled"}
          </span>
          <Switch
            checked={enabled}
            onChange={setEnabled}
            className={`${enabled ? "bg-blue-600" : "bg-gray-200"} relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500`}
          >
            <span
              className={`${enabled ? "translate-x-6" : "translate-x-1"} inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
            />
          </Switch>
        </div>
      </div>

      <div
        className={`p-6 grid grid-cols-1 lg:grid-cols-2 gap-8 ${!enabled ? "opacity-50 pointer-events-none grayscale-[0.5]" : ""}`}
      >
        {/* Left Column: Schedule Setup */}
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Frequency
            </label>
            <select
              value={frequency}
              onChange={(e) => setFrequency(e.target.value as any)}
              className="w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="weekly">Weekly (1x / week)</option>
              <option value="biweekly">Bi-weekly (2x / week)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Execution Time (WIB UTC+7)
            </label>
            <div className="flex items-center space-x-2">
              <select
                value={executionTime.split(":")[0] || "02"}
                onChange={(e) =>
                  setExecutionTime(
                    `${e.target.value}:${executionTime.split(":")[1] || "00"}`,
                  )
                }
                className="border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              >
                {Array.from({ length: 24 }).map((_, i) => {
                  const hour = i.toString().padStart(2, "0");
                  return (
                    <option key={hour} value={hour}>
                      {hour}
                    </option>
                  );
                })}
              </select>
              <span className="font-bold">:</span>
              <select
                value={executionTime.split(":")[1] || "00"}
                onChange={(e) =>
                  setExecutionTime(
                    `${executionTime.split(":")[0] || "02"}:${e.target.value}`,
                  )
                }
                className="border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              >
                {Array.from({ length: 60 }).map((_, i) => {
                  const min = i.toString().padStart(2, "0");
                  return (
                    <option key={min} value={min}>
                      {min}
                    </option>
                  );
                })}
              </select>
            </div>
          </div>

          {frequency === "weekly" && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Run on
              </label>
              <select
                value={dayOfWeek}
                onChange={(e) => setDayOfWeek(Number(e.target.value))}
                className="w-full border-gray-300 rounded-md shadow-sm"
              >
                {DAYS.map((d, i) => (
                  <option key={i} value={i}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          )}

          {frequency === "biweekly" && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Run on (Select exactly 2)
              </label>
              <div className="flex gap-2 flex-wrap">
                {DAYS.map((d, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => toggleArrayItem(setDaysOfWeek, i)}
                    className={`px-3 py-1 text-sm rounded ${daysOfWeek.includes(i) ? "bg-blue-100 text-blue-700 border border-blue-300" : "bg-gray-100 text-gray-600 border border-gray-200"}`}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Dataset Checkboxes */}
          <div className="pt-4 border-t">
            <h3 className="font-semibold text-gray-800 mb-3">
              Datasets to Auto Process
            </h3>

            {[
              {
                key: "nasa_refresh",
                label: "NASA Refresh",
                list: availableDatasets.nasa,
              },
              {
                key: "nasa_preprocess",
                label: "NASA Preprocess",
                list: availableDatasets.nasa,
              },
              {
                key: "bmkg_preprocess",
                label: "BMKG Preprocess",
                list: availableDatasets.bmkg,
              },
            ].map((cat) => {
              const categoryKey = cat.key as keyof typeof selectedDatasets;
              const lists = cat.list;
              return (
                <div key={categoryKey} className="mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium uppercase text-gray-600">
                      {cat.label}
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        selectAll(
                          categoryKey,
                          selectedDatasets[categoryKey].length !== lists.length,
                          lists,
                        )
                      }
                      className="text-xs text-blue-600"
                    >
                      {selectedDatasets[categoryKey].length === lists.length
                        ? "Deselect All"
                        : "Select All"}
                    </button>
                  </div>
                  {lists.length === 0 ? (
                    <p className="text-xs text-gray-400 italic mb-2">
                      No raw datasets available
                    </p>
                  ) : (
                    <div className="max-h-32 overflow-y-auto space-y-1 bg-gray-50 p-2 rounded border">
                      {lists.map((ds) => (
                        <label
                          key={ds.collectionName}
                          className="flex items-center text-sm cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={selectedDatasets[categoryKey].includes(
                              ds.collectionName,
                            )}
                            onChange={() =>
                              toggleDataset(categoryKey, ds.collectionName)
                            }
                            className="rounded border-gray-300 text-blue-600 mr-2"
                          />
                          {ds.collectionName}
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Previews and Recommendations */}
        <div>
          <div className="bg-blue-50/50 border border-blue-100 p-4 rounded-xl mb-4">
            <h3 className="text-sm font-semibold text-blue-800 mb-3">
              Next 5 Scheduled Runs
            </h3>
            {!enabled ? (
              <p className="text-sm text-gray-500 italic">
                Automation disabled
              </p>
            ) : nextRunsPreview.length === 0 ? (
              <p className="text-sm text-gray-500 italic">Calculating...</p>
            ) : (
              <ul className="space-y-2 text-sm text-gray-700">
                {nextRunsPreview.map((iso, i) => {
                  const d = new Date(iso);
                  return (
                    <li
                      key={i}
                      className="flex items-center before:content-['•'] before:mr-2 before:text-blue-400"
                    >
                      {d.toLocaleString("en-US", {
                        weekday: "short",
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}{" "}
                      WIB
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="bg-orange-50 border border-orange-100 p-4 rounded-xl text-sm text-orange-800">
            <div className="flex font-semibold items-center mb-2">
              <AlertTriangle className="w-4 h-4 mr-2" /> Recommendations
            </div>
            <ul className="space-y-1 mb-2 ml-6 list-disc">
              <li>
                <b>Weekly:</b> Best for stable long-term data sources.
              </li>
              <li>
                <b>Bi-weekly:</b> Balanced between freshness & server load.
              </li>
            </ul>
            <p className="mt-3 text-orange-700 italic text-xs">
              Note: Executing massive dataset preprocesses too frequently can
              negatively impact system resources.
            </p>
          </div>
        </div>
      </div>

      <div className="p-4 bg-gray-50 border-t flex justify-end items-center gap-3">
        <button
          onClick={loadData}
          disabled={saving}
          className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg"
        >
          Reset
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center"
        >
          {saving ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Save className="w-4 h-4 mr-2" />
          )}
          Save Configuration
        </button>
      </div>
    </div>
  );
}
