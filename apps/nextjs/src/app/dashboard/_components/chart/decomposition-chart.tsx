"use client";

import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
  Area,
  AreaChart,
  Scatter,
  ScatterChart,
  ReferenceLine,
} from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Icons } from "@/app/dashboard/_components/icons";
import { getDecompositionByPreprocessingId } from "@/lib/fetch/files.fetch";
import { useMemo, useState, useTransition } from "react";
import * as downsample from "downsample-lttb";

interface ChartDataPoint {
  date: number;
  fullDate: string;
  original: number;
  trend: number | null;
  seasonal: number | null;
  residual: number | null;
  year: number;
}

interface ComponentStats {
  min: number;
  max: number;
}

interface DecompositionStats {
  trend: ComponentStats;
  seasonal: ComponentStats;
  residual: ComponentStats;
}

interface DecompositionChartProps {
  preprocessingId: string;
}

interface ParameterDecompositionViewProps {
  param: string;
  paramData: any;
  decompositionMethod: string;
}

// CONFIGURATION & CONSTANTS
const chartConfig = {
  original: {
    label: "Original",
    color: "hsl(var(--chart-1))",
  },
  trend: {
    label: "Trend",
    color: "#2563eb",
  },
  seasonal: {
    label: "Seasonal",
    color: "#2563eb",
  },
  residual: {
    label: "Residual",
    color: "#2563eb",
  },
} satisfies ChartConfig;

const paramLabels: Record<string, string> = {
  // NASA
  T2M: "Suhu Udara (2m)",
  T2M_MAX: "Suhu Maksimum",
  T2M_MIN: "Suhu Minimum",
  RH2M: "Kelembaban Udara",
  PRECTOTCORR: "Curah Hujan",
  ALLSKY_SFC_SW_DWN: "Radiasi Matahari",
  WS10M: "Kecepatan Angin",
  WS10M_MAX: "Kecepatan Angin Maksimum",
  WD10M: "Arah Angin",
  // BMKG
  TN: "Temperatur Minimum",
  TX: "Temperatur Maksimum",
  TAVG: "Temperatur Rata-rata",
  RH_AVG: "Kelembapan Rata-rata",
  RR: "Curah Hujan",
  SS: "Lamanya Penyinaran Matahari",
  FF_X: "Kecepatan Angin Maksimum",
  DDD_X: "Arah Angin saat Kecepatan Maksimum",
  FF_AVG: "Kecepatan Angin Rata-rata",
  DDD_CAR: "Arah Angin Terbanyak",
};

const paramUnits: Record<string, string> = {
  // NASA
  T2M: "°C",
  T2M_MAX: "°C",
  T2M_MIN: "°C",
  RH2M: "%",
  PRECTOTCORR: "mm",
  ALLSKY_SFC_SW_DWN: "MJ/m²/day",
  WS10M: "m/s",
  WS10M_MAX: "m/s",
  WD10M: "degrees",
  // BMKG
  TN: "°C",
  TX: "°C",
  TAVG: "°C",
  RH_AVG: "%",
  RR: "mm",
  SS: "jam",
  FF_X: "m/s",
  DDD_X: "derajat",
  FF_AVG: "m/s",
  DDD_CAR: "°",
};

// UTILITY FUNCTIONS
function getParamLabel(param: string): string {
  return paramLabels[param] || param;
}

function getParamUnit(param: string): string {
  return paramUnits[param] || "";
}

/**
 * Calculate custom ticks array for exactly every 4 years
 */
function calculateYearTicks(chartData: ChartDataPoint[]): number[] {
  if (!chartData.length) return [];
  const years = chartData.map((d) => d.year);
  const minYear = Math.min(...years);
  const maxYear = Math.max(...years);

  const ticks: number[] = [];
  const startYear = Math.ceil(minYear / 4) * 4;
  for (let y = startYear; y <= maxYear; y += 4) {
    // using UTC to avoid any timezone shift rendering issues on first day of year
    ticks.push(Date.UTC(y, 0, 1));
  }
  return ticks;
}

// UI COMPONENTS

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-10 w-64" />
      <div className="flex gap-2">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-9 w-32 rounded-lg" />
        ))}
      </div>
      {[...Array(4)].map((_, i) => (
        <Card key={i}>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[220px] w-full rounded-lg" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function EmptyState({ message }: { message?: string }) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-16">
        <Icons.activity className="h-12 w-12 text-muted-foreground mb-4" />
        <p className="font-medium text-lg">
          {message || "Tidak ada data dekomposisi"}
        </p>
        <p className="text-sm text-muted-foreground text-center max-w-sm mt-1">
          Pastikan dataset telah dipreprocessing terlebih dahulu
        </p>
      </CardContent>
    </Card>
  );
}

function ParameterDecompositionView({
  param,
  paramData,
  decompositionMethod,
}: ParameterDecompositionViewProps) {
  // Data mapping
  const chartData: ChartDataPoint[] = useMemo(() => {
    if (!paramData?.data || !Array.isArray(paramData.data)) return [];

    // 1. Map all data points first
    const fullData: ChartDataPoint[] = paramData.data.map((item: any) => {
      const rawDate = item.Date?.$date || item.Date;
      const dateObj = new Date(rawDate);
      const startOfDayUTC = Date.UTC(
        dateObj.getUTCFullYear(),
        dateObj.getUTCMonth(),
        dateObj.getUTCDate(),
      );

      return {
        date: startOfDayUTC,
        fullDate: dateObj.toLocaleDateString("id-ID", {
          weekday: "long",
          year: "numeric",
          month: "long",
          day: "numeric",
        }),
        original: item.original,
        trend: item.trend,
        seasonal: item.seasonal,
        residual: item.residual,
        year: dateObj.getFullYear(),
      };
    });

    // 2. Determine if we need to downsample
    const TARGET_RESOLUTION = 500;
    if (fullData.length <= TARGET_RESOLUTION) {
      return fullData;
    }

    // 3. Prepare data for LTTB
    // FIX: explicitly type 'd' as ChartDataPoint
    const lttbInput: [number, number][] = fullData.map((d: ChartDataPoint) => [
      d.date,
      d.original ?? 0,
    ]);

    // 4. Run LTTB algorithm
    const downsampledPairs = downsample.processData(
      lttbInput,
      TARGET_RESOLUTION,
    );

    // 5. Create a Set of timestamps for O(1) fast lookup
    // FIX: explicitly type 'pair' as [number, number]
    const preservedTimestamps = new Set(
      downsampledPairs.map((pair: [number, number]) => pair[0]),
    );

    // 6. Filter the full dataset
    // FIX: explicitly type 'd' as ChartDataPoint
    return fullData.filter((d: ChartDataPoint) =>
      preservedTimestamps.has(d.date),
    );
  }, [paramData]);

  const stats = useMemo<DecompositionStats | null>(() => {
    if (!chartData.length) return null;

    const filterNulls = (arr: (number | null)[]): number[] =>
      arr.filter((v): v is number => v !== null && !isNaN(v));

    const trends = filterNulls(chartData.map((d) => d.trend));
    const seasonals = filterNulls(chartData.map((d) => d.seasonal));
    const residuals = filterNulls(chartData.map((d) => d.residual));

    const safelyGetMinMax = (arr: number[]): ComponentStats => {
      if (arr.length === 0) return { min: 0, max: 0 };
      return { min: Math.min(...arr), max: Math.max(...arr) };
    };

    return {
      trend: safelyGetMinMax(trends),
      seasonal: safelyGetMinMax(seasonals),
      residual: safelyGetMinMax(residuals),
    };
  }, [chartData]);

  // Skeleton while loading or if data is not ready
  if (!chartData.length || !stats) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Memproses {getParamLabel(param)}...</CardTitle>
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[220px] w-full rounded-lg" />
          </CardContent>
        </Card>
      </div>
    );
  }

  // UI & metadata for decomposition view
  const dateRange = `${chartData[0]?.fullDate} - ${chartData[chartData.length - 1]?.fullDate}`;
  const xAxisTicks = calculateYearTicks(chartData);
  const unit = getParamUnit(param);

  return (
    <div className="space-y-4">
      {/* METADATA CARDS */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Metode</CardTitle>
            <Icons.activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{decompositionMethod}</div>
            <p className="text-xs text-muted-foreground mt-1">Dekomposisi</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Seasonal Strength
            </CardTitle>
            <Icons.zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <div className="text-2xl font-bold">
                {paramData.seasonal_strength?.toFixed(3) || "N/A"}
              </div>
              {paramData.seasonal_strength > 0.5 && (
                <Badge variant="secondary" className="text-[10px]">
                  Kuat
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Pengaruh musiman
            </p>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Rentang Waktu</CardTitle>
            <Icons.calendar className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold">{dateRange}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Total {chartData.length.toLocaleString("id-ID")} hari observasi
            </p>
          </CardContent>
        </Card>
      </div>

      {/* CHART 1: ORIGINAL */}
      <Card>
        <CardHeader className="py-3">
          <div className="flex items-center gap-2">
            <Icons.activity className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-sm">Data Original</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="pb-4">
          <ChartContainer
            config={chartConfig}
            style={{ height: "150px", width: "100%" }}
          >
            <LineChart
              data={chartData}
              syncId="decomposition-sync"
              margin={{ top: 5, right: 10, left: 10, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="date"
                type="number"
                scale="time"
                domain={["dataMin", "dataMax"]}
                ticks={xAxisTicks}
                tickFormatter={(val) =>
                  new Date(val).getUTCFullYear().toString()
                }
                tickLine={false}
                axisLine={false}
                tickMargin={8}
              />
              <YAxis
                domain={["auto", "auto"]}
                width={50}
                tickFormatter={(val) => `${val}${unit}`}
                type="number"
                allowDataOverflow
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_, payload) =>
                      payload[0]?.payload?.fullDate || ""
                    }
                  />
                }
              />
              <Line
                type="monotone"
                dataKey="original"
                stroke="hsl(var(--chart-1))"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* DECOMPOSITION COMPONENTS */}
      <Card>
        <CardHeader className="py-4">
          <CardTitle className="text-sm border-b pb-2 mb-2">
            Komponen Dekomposisi
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4 space-y-6">
          {/* CHART 2: TREND */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Icons.trendingUp className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-semibold">Trend</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Min: {stats.trend.min.toFixed(2)} | Max:{" "}
                {stats.trend.max.toFixed(2)}
              </div>
            </div>
            <ChartContainer
              config={chartConfig}
              style={{ height: "150px", width: "100%" }}
            >
              <LineChart
                data={chartData}
                syncId="decomposition-sync"
                margin={{ top: 5, right: 10, left: 10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="date"
                  type="number"
                  scale="time"
                  domain={["dataMin", "dataMax"]}
                  ticks={xAxisTicks}
                  tickFormatter={(val) =>
                    new Date(val).getUTCFullYear().toString()
                  }
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                />
                <YAxis
                  domain={["auto", "auto"]}
                  width={50}
                  tickFormatter={(val) => `${val}${unit}`}
                  type="number"
                  allowDataOverflow
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      labelFormatter={(_, payload) =>
                        payload[0]?.payload?.fullDate || ""
                      }
                    />
                  }
                />
                <Line
                  type="monotone"
                  dataKey="trend"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ChartContainer>
          </div>

          <div className="border-t border-border" />

          {/* CHART 3: SEASONAL */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Icons.waves className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-semibold">Seasonal</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Max Var: ±
                {Math.max(
                  Math.abs(stats.seasonal.min),
                  Math.abs(stats.seasonal.max),
                ).toFixed(2)}{" "}
                {unit}
              </div>
            </div>
            <ChartContainer
              config={chartConfig}
              style={{ height: "150px", width: "100%" }}
            >
              <AreaChart
                data={chartData}
                syncId="decomposition-sync"
                margin={{ top: 5, right: 10, left: 10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="date"
                  type="number"
                  scale="time"
                  domain={["dataMin", "dataMax"]}
                  ticks={xAxisTicks}
                  tickFormatter={(val) =>
                    new Date(val).getUTCFullYear().toString()
                  }
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                />
                <YAxis
                  domain={["auto", "auto"]}
                  width={50}
                  tickFormatter={(val) => `${val}${unit}`}
                  type="number"
                  allowDataOverflow
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      labelFormatter={(_, payload) =>
                        payload[0]?.payload?.fullDate || ""
                      }
                    />
                  }
                />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                <Area
                  type="monotone"
                  dataKey="seasonal"
                  stroke="#2563eb"
                  fill="#2563eb"
                  fillOpacity={0.3}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ChartContainer>
          </div>

          <div className="border-t border-border" />

          {/* CHART 4: RESIDUAL */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Icons.activity className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-semibold">Residual</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Noise Bound: {stats.residual.min.toFixed(2)} to{" "}
                {stats.residual.max.toFixed(2)}
              </div>
            </div>
            <ChartContainer
              config={chartConfig}
              style={{ height: "150px", width: "100%" }}
            >
              <LineChart
                data={chartData}
                syncId="decomposition-sync"
                margin={{ top: 5, right: 10, left: 10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="date"
                  type="number"
                  scale="time"
                  domain={["dataMin", "dataMax"]}
                  ticks={xAxisTicks}
                  tickFormatter={(val) =>
                    new Date(val).getUTCFullYear().toString()
                  }
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                />
                <YAxis
                  domain={["auto", "auto"]}
                  width={50}
                  tickFormatter={(val) => `${val}${unit}`}
                  type="number"
                  allowDataOverflow
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      labelFormatter={(_, payload) =>
                        payload[0]?.payload?.fullDate || ""
                      }
                    />
                  }
                />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                {/* Using Line with dot=true instead of Scatter for more reliable time-series plotting */}
                <Line
                  type="monotone"
                  dataKey="residual"
                  stroke="#2563eb"
                  strokeWidth={0}
                  dot={{ r: 1, fill: "#2563eb" }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ChartContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export function DecompositionChart({
  preprocessingId,
}: DecompositionChartProps) {
  const [selectedParam, setSelectedParam] = useState<string>("");
  const [isPending, startTransition] = useTransition(); // Add this

  // PHASE 2: Fetch the entire decomposition report using the preprocessingId
  const {
    data: report,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["decomposition-report", preprocessingId],
    queryFn: () => getDecompositionByPreprocessingId(preprocessingId),
    refetchOnWindowFocus: false,
  });

  if (isLoading) {
    return <LoadingSkeleton />;
  }

  if (
    error ||
    !report ||
    !report.parameters ||
    Object.keys(report.parameters).length === 0
  ) {
    return (
      <EmptyState message="Tidak ada data dekomposisi yang tersedia untuk dataset ini" />
    );
  }

  // PHASE 2: Extract available parameters from the report keys
  const availableParams = Object.keys(report.parameters);
  const activeParam = selectedParam || availableParams[0];

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold tracking-tight">
          Analisis Dekomposisi
        </h2>
        <p className="text-muted-foreground">
          Visualisasi dekomposisi time series untuk memahami tren, komponen
          musiman (seasonal), dan residual.
        </p>
      </div>

      <div className="flex flex-col gap-6 pt-2">
        <div className="flex justify-start sm:justify-end">
          {/* Use startTransition here */}
          <Select
            value={activeParam}
            onValueChange={(val) =>
              startTransition(() => setSelectedParam(val))
            }
          >
            <SelectTrigger className="w-full sm:w-[250px]">
              <SelectValue placeholder="Pilih parameter" />
            </SelectTrigger>
            <SelectContent>
              {availableParams.map((param) => (
                <SelectItem key={param} value={param}>
                  <div className="flex items-center justify-between gap-2">
                    <span>{getParamLabel(param)}</span>
                    <span className="text-xs text-muted-foreground">
                      ({param})
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div
          className={
            isPending ? "opacity-50 transition-opacity" : "transition-opacity"
          }
        >
          <ParameterDecompositionView
            param={activeParam}
            paramData={report.parameters[activeParam]}
            decompositionMethod={report.decomposition_method}
          />
        </div>
      </div>
    </div>
  );
}
