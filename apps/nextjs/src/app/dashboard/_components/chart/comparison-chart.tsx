"use client";
import { GetComparisonData, GetChartDataBySlug } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis, Brush } from "recharts";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Icons } from "@/app/dashboard/_components/icons";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import * as downsample from "downsample-lttb";
import WindRoseChart from "./wind-rose-chart";

//LTTB downsampling
const TARGET_POINTS = 300;

interface ComparisonChartProps {
  originalCollectionName: string;
  cleanedCollectionName: string;
}

interface RawDailyRecord {
  dateTimestamp: number;
  dateString: string;
  original: number | null;
  cleaned: number | null;
}

const paramLabels: Record<string, string> = {
  T2M: "Suhu Udara (2m)",
  T2M_MAX: "Suhu Maksimum (2m)",
  T2M_MIN: "Suhu Minimum (2m)",
  RH2M: "Kelembaban Relatif (2m)",
  PRECTOTCORR: "Curah Hujan",
  ALLSKY_SFC_SW_DWN: "Radiasi Matahari",
  WS10M: "Kecepatan Angin (10m)",
  WS10M_MAX: "Kecepatan Angin Maksimum (10m)",
  WD10M: "Arah Angin (10m)",
};

const paramUnits: Record<string, string> = {
  T2M: "°C",
  T2M_MAX: "°C",
  T2M_MIN: "°C",
  RH2M: "%",
  PRECTOTCORR: "mm/hari",
  ALLSKY_SFC_SW_DWN: "MJ/m²/hari",
  WS10M: "m/s",
  WS10M_MAX: "m/s",
  WD10M: "°",
};

const windDirectionSpeedMap: Record<string, string> = {
  WD10M: "WS10M",
  DDD_CAR: "FF_AVG",
  DDD_X: "FF_X",
};

export default function ComparisonChart({
  originalCollectionName,
  cleanedCollectionName,
}: ComparisonChartProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: [
      "comparison-data",
      originalCollectionName,
      cleanedCollectionName,
    ],
    queryFn: () =>
      GetComparisonData(originalCollectionName, cleanedCollectionName),
    staleTime: 5 * 60 * 1000, // Cache 5 minutes
  });

  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [brushDomain, setBrushDomain] = useState<{
    start: number;
    end: number;
  } | null>(null);

  // State flags wind direction
  const isWindDirection = Object.keys(windDirectionSpeedMap).includes(
    selectedColumn,
  );
  const pairedSpeedColumn = isWindDirection
    ? windDirectionSpeedMap[selectedColumn]
    : null;

  // Fetch full dataset when a wind parameter is selected
  const { data: originalFullData, isLoading: isLoadingOriginal } = useQuery({
    queryKey: ["chart-data", originalCollectionName],
    queryFn: () => GetChartDataBySlug(originalCollectionName),
    enabled: isWindDirection, // Only fetch if wind direction is selected
    staleTime: 5 * 60 * 1000,
  });

  const { data: cleanedFullData, isLoading: isLoadingCleaned } = useQuery({
    queryKey: ["chart-data", cleanedCollectionName],
    queryFn: () => GetChartDataBySlug(cleanedCollectionName),
    enabled: isWindDirection, // Only fetch if wind direction is selected
    staleTime: 5 * 60 * 1000,
  });

  // Raw daily data preparation
  const rawDailyData = useMemo(() => {
    if (!data?.differences?.length || !selectedColumn) {
      return [];
    }
    console.log(`Formatting ${data.differences.length} raw daily rows...`);
    return data.differences
      .map((item: any) => {
        const dateObj = new Date(item.date);
        const origRaw = item[selectedColumn]?.original;
        const cleanRaw = item[selectedColumn]?.cleaned;

        // Ensure invalid values (8888/9999) break the line as null
        const isInvalid = (val: any) =>
          val === 8888 ||
          val === 9999 ||
          val === null ||
          val === undefined ||
          val === "None" ||
          val === "NaN";

        return {
          dateTimestamp: dateObj.getTime(),
          dateString: item.date,
          original: isInvalid(origRaw) ? null : origRaw,
          cleaned: isInvalid(cleanRaw) ? null : cleanRaw,
        } as RawDailyRecord;
      })
      .sort(
        (a: RawDailyRecord, b: RawDailyRecord) =>
          a.dateTimestamp - b.dateTimestamp,
      );
  }, [data?.differences, selectedColumn]);

  const lttbData = useMemo(() => {
    // Bypass LTTB calculation entirely if it's a wind direction
    if (isWindDirection) return [];
    if (!rawDailyData.length) return [];

    // If the data is already small, skip LTTB
    if (rawDailyData.length <= TARGET_POINTS) return rawDailyData;

    console.log("Applying LTTB downsampling...");
    // 1. Prepare data for LTTB (Array of [x, y] tuples) - ignore null values
    const origSeries = rawDailyData
      .filter((d: RawDailyRecord) => d.original !== null)
      .map(
        (d: RawDailyRecord) =>
          [d.dateTimestamp, d.original] as [number, number],
      );

    const cleanSeries = rawDailyData
      .filter((d: RawDailyRecord) => d.cleaned !== null)
      .map(
        (d: RawDailyRecord) => [d.dateTimestamp, d.cleaned] as [number, number],
      );

    // 2. Execute LTTB
    const sampledOrig: [number, number][] =
      origSeries.length > 0
        ? downsample.processData(
            origSeries,
            Math.min(TARGET_POINTS, origSeries.length),
          )
        : [];

    const sampledClean: [number, number][] =
      cleanSeries.length > 0
        ? downsample.processData(
            cleanSeries,
            Math.min(TARGET_POINTS, cleanSeries.length),
          )
        : [];

    // MICRO-OPTIMIZATION: Convert array to Map for O(1) lookup
    const origMap = new Map(sampledOrig);
    const cleanMap = new Map(sampledClean);

    // 3. Merge Back
    // Gather all unique X coordinates (timestamps) that LTTB kept
    const timestampSet = new Set<number>();
    sampledOrig.forEach((d: [number, number]) => timestampSet.add(d[0]));
    sampledClean.forEach((d: [number, number]) => timestampSet.add(d[0]));

    const sortedTimestamps = Array.from(timestampSet).sort((a, b) => a - b);

    // Dictionary for fast lookup of original properties (like dateString)
    const rawDict = new Map<number, RawDailyRecord>();
    rawDailyData.forEach((d: RawDailyRecord) =>
      rawDict.set(d.dateTimestamp, d),
    );

    // Construct the merged array
    return sortedTimestamps.map((ts) => {
      const rawRef = rawDict.get(ts);
      // MICRO-OPTIMIZATION: Use Map.get() instead of Array.find()
      const origMatch = origMap.get(ts);
      const cleanMatch = cleanMap.get(ts);

      return {
        dateTimestamp: ts,
        dateString:
          rawRef?.dateString || new Date(ts).toISOString().split("T")[0],
        original: origMatch !== undefined ? origMatch : null,
        cleaned: cleanMatch !== undefined ? cleanMatch : null,
      } as RawDailyRecord;
    });
  }, [rawDailyData, isWindDirection]);

  const displayData = useMemo(() => {
    // 1. If zoomed out entirely, return LTTB Overview
    if (!brushDomain) return lttbData;

    // 2. Slice the raw daily data based on the Brush's timestamp domain
    const subset = rawDailyData.filter(
      (d: RawDailyRecord) =>
        d.dateTimestamp >= brushDomain.start &&
        d.dateTimestamp <= brushDomain.end,
    );

    // 3. If the selection is narrow enough (< 300 days), show exact raw daily data!
    if (subset.length > 0 && subset.length <= TARGET_POINTS) {
      return subset;
    }

    // 4. If the selection is still huge (> 300 days), just slice the LTTB array to save performance
    return lttbData.filter(
      (d: RawDailyRecord) =>
        d.dateTimestamp >= brushDomain.start &&
        d.dateTimestamp <= brushDomain.end,
    );
  }, [brushDomain, rawDailyData, lttbData]);

  const isDailyMode =
    brushDomain &&
    rawDailyData.filter(
      (d: RawDailyRecord) =>
        d.dateTimestamp >= brushDomain.start &&
        d.dateTimestamp <= brushDomain.end,
    ).length <= TARGET_POINTS;

  const chartData = lttbData;

  // Data transformation for WindRoses
  const windRoseDataSets = useMemo(() => {
    if (!isWindDirection || !pairedSpeedColumn) {
      return { original: [], cleaned: [] };
    }

    const original: any[] = [];
    const cleaned: any[] = [];

    const isInvalid = (val: any) =>
      val === 8888 ||
      val === 9999 ||
      val === null ||
      val === undefined ||
      val === "None" ||
      val === "NaN";

    // 1. Extract original valid pairs from the full original dataset
    if (originalFullData?.items) {
      originalFullData.items.forEach((item: any) => {
        const origDir = item[selectedColumn];
        const origSpeed = item[pairedSpeedColumn];

        if (!isInvalid(origDir) && !isInvalid(origSpeed)) {
          original.push({
            [selectedColumn]: origDir,
            [pairedSpeedColumn]: origSpeed,
          });
        }
      });
    }

    // 2. Extract cleaned valid pairs from the full cleaned dataset
    if (cleanedFullData?.items) {
      cleanedFullData.items.forEach((item: any) => {
        const cleanDir = item[selectedColumn];
        const cleanSpeed = item[pairedSpeedColumn];

        if (!isInvalid(cleanDir) && !isInvalid(cleanSpeed)) {
          cleaned.push({
            [selectedColumn]: cleanDir,
            [pairedSpeedColumn]: cleanSpeed,
          });
        }
      });
    }
    return { original, cleaned };
  }, [
    originalFullData,
    cleanedFullData,
    isWindDirection,
    selectedColumn,
    pairedSpeedColumn,
  ]);

  // Auto select first column when data loads
  useMemo(() => {
    if (data?.summary?.numericColumns?.length && !selectedColumn) {
      setSelectedColumn(data.summary.numericColumns[0]);
    }
  }, [data, selectedColumn]);

  const isAnyLoading =
    isLoading || (isWindDirection && (isLoadingOriginal || isLoadingCleaned));

  // Loading state
  if (isAnyLoading) {
    return (
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Icons.spinner className="h-5 w-5 animate-spin" />
            Memuat Data...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-muted-foreground">
              Sedang memproses {originalCollectionName}...
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }
  // Error state
  if (error) {
    return (
      <Alert variant="destructive" className="mt-6">
        <Icons.alert className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>Gagal memuat data perbandingan.</AlertDescription>
      </Alert>
    );
  }
  // No data state
  if (!data || !data.differences?.length) {
    return (
      <Alert className="mt-6 border-yellow-500">
        <Icons.alert className="h-4 w-4 text-yellow-600" />
        <AlertTitle className="text-yellow-600">Data Tidak Tersedia</AlertTitle>
        <AlertDescription>
          Tidak ada data perbandingan untuk ditampilkan.
        </AlertDescription>
      </Alert>
    );
  }
  const chartConfig = {
    original: {
      label: "Original",
      color: "hsl(var(--chart-1))",
    },
    cleaned: {
      label: "Cleaned",
      color: "hsl(var(--chart-2))",
    },
  } satisfies ChartConfig;
  return (
    <Card className="mt-6">
      <CardHeader>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Icons.gitCompare className="h-5 w-5" />
              Perbandingan: {paramLabels[selectedColumn] || selectedColumn}
            </CardTitle>
            <CardDescription className="mt-2">
              Original vs Cleaned Dataset (Data Harian)
            </CardDescription>
            <div className="flex gap-2 pt-2 flex-wrap">
              <Badge variant="outline" className="text-xs">
                Parameter: {selectedColumn}
              </Badge>
              {paramUnits[selectedColumn] && (
                <Badge variant="outline" className="text-xs">
                  Unit: {paramUnits[selectedColumn]}
                </Badge>
              )}
              {/* Visual Indicators */}
              <Badge
                variant="outline"
                className={`text-xs ${isDailyMode ? "bg-green-100 text-green-800 border-green-300" : "bg-blue-100 text-blue-800 border-blue-300"}`}
              >
                {isDailyMode ? "Mode: Detail Harian" : "Mode: Overview (LTTB)"}
              </Badge>
              <Badge variant="outline" className="text-xs bg-muted">
                {rawDailyData.length} total hari
              </Badge>
            </div>
          </div>

          {/* COLUMN SELECTOR */}
          <Select value={selectedColumn} onValueChange={setSelectedColumn}>
            <SelectTrigger className="w-full sm:w-[250px]">
              <SelectValue placeholder="Pilih parameter" />
            </SelectTrigger>
            <SelectContent>
              {data.summary.numericColumns.map((column: string) => (
                <SelectItem key={column} value={column}>
                  <div className="flex items-center justify-between gap-2">
                    <span>{paramLabels[column] || column}</span>
                    <span className="text-xs text-muted-foreground">
                      ({column})
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-2">
        {/* Mengecek apakah parameter yang dipilih adalah arah angin dan memiliki pasangan kolom kecepatan */}
        {isWindDirection && pairedSpeedColumn ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
            <WindRoseChart
              data={windRoseDataSets.original}
              directionColumn={selectedColumn}
              speedColumn={pairedSpeedColumn}
              title="Dataset Original"
              domainMax="auto"
            />
            <WindRoseChart
              data={windRoseDataSets.cleaned}
              directionColumn={selectedColumn}
              speedColumn={pairedSpeedColumn}
              title="Dataset Cleaned"
              domainMax="auto"
            />
          </div>
        ) : (
          <>
            {/* Main Detail Chart (Line Chart untuk parameter non-angin) */}
            <ChartContainer config={chartConfig}>
              <LineChart
                accessibilityLayer
                data={displayData}
                margin={{ left: 12, right: 12, top: 10, bottom: 5 }}
              >
                <CartesianGrid vertical={false} strokeDasharray="3 3" />

                <XAxis
                  dataKey="dateString"
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  minTickGap={32}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    if (isDailyMode) {
                      return date.toLocaleDateString("id-ID", {
                        day: "numeric",
                        month: "short",
                        year: "2-digit",
                      });
                    }
                    return date.toLocaleDateString("id-ID", {
                      month: "short",
                      year: "numeric",
                    });
                  }}
                />

                <YAxis
                  domain={["auto", "auto"]}
                  tickFormatter={(value) => {
                    const unit = paramUnits[selectedColumn] || "";
                    return `${value.toFixed(1)}${unit}`;
                  }}
                />

                <ChartTooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={
                    <ChartTooltipContent
                      labelFormatter={(label) => {
                        const date = new Date(label);
                        const formattedDate = date.toLocaleDateString("id-ID", {
                          day: "numeric",
                          month: "long",
                          year: "numeric",
                        });
                        return isDailyMode
                          ? formattedDate
                          : `${formattedDate} (Estimasi Tren)`;
                      }}
                      formatter={(value, name) => {
                        const unit = paramUnits[selectedColumn] || "";
                        const labelName =
                          name === "original" ? "Original" : "Cleaned";
                        const displayValue = isDailyMode
                          ? `${Number(value).toFixed(2)} ${unit}`
                          : `~${Number(value).toFixed(2)} ${unit}`;
                        return [displayValue, labelName];
                      }}
                    />
                  }
                />

                <Line
                  dataKey="original"
                  type="monotone"
                  stroke="hsl(var(--chart-1))"
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                />

                <Line
                  dataKey="cleaned"
                  type="monotone"
                  stroke="hsl(var(--chart-2))"
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                />
              </LineChart>
            </ChartContainer>

            {/* Overview Chart with Brush (Mini chart untuk navigasi timeline) */}
            {lttbData.length > 0 && (
              <ChartContainer
                config={chartConfig}
                style={{ minHeight: "80px", height: "80px" }}
              >
                <LineChart data={lttbData} margin={{ left: 12, right: 12 }}>
                  <Brush
                    dataKey="dateString"
                    height={30}
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--muted))"
                    tickFormatter={(val) =>
                      new Date(val).getFullYear().toString()
                    }
                    onChange={(e) => {
                      if (
                        e.startIndex !== undefined &&
                        e.endIndex !== undefined
                      ) {
                        if (
                          e.startIndex === 0 &&
                          e.endIndex === lttbData.length - 1
                        ) {
                          setBrushDomain(null);
                        } else {
                          const startTs = lttbData[e.startIndex]?.dateTimestamp;
                          const endTs = lttbData[e.endIndex]?.dateTimestamp;
                          if (startTs && endTs) {
                            setBrushDomain({ start: startTs, end: endTs });
                          }
                        }
                      }
                    }}
                  />
                </LineChart>
              </ChartContainer>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
