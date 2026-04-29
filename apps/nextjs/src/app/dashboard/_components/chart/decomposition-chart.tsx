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
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Icons } from "@/app/dashboard/_components/icons";
import { getDecompositionByPreprocessingId } from "@/lib/fetch/files.fetch";
import { useMemo } from "react";

interface ChartDataPoint {
  date: string;
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
    color: "hsl(var(--chart-2))",
  },
  seasonal: {
    label: "Seasonal",
    color: "hsl(var(--chart-3))",
  },
  residual: {
    label: "Residual",
    color: "hsl(var(--chart-4))",
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
 * Calculate optimal tick interval based on year range
 */
function calculateTickInterval(chartData: ChartDataPoint[]): number {
  const years = [...new Set(chartData.map((d) => d.year))].sort();
  const yearRange = years.length;

  if (yearRange > 20) {
    return Math.ceil(chartData.length / 10);
  } else if (yearRange > 10) {
    return Math.ceil(chartData.length / 15);
  } else {
    return Math.ceil(chartData.length / 20);
  }
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

    return paramData.data.map((item: any) => {
      // Unwrapping MongoDB $date if present, else fallback to standard string
      const rawDate = item.Date?.$date || item.Date;
      const dateObj = new Date(rawDate);

      return {
        date: dateObj.getFullYear().toString(),
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
  }, [paramData]);

  // Calculate stats for min/max  boundaries
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
  const tickInterval = calculateTickInterval(chartData);
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
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
                interval={tickInterval}
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
                stroke="var(--color-original)"
                strokeWidth={1}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* CHART 2: TREND */}
      <Card>
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Icons.trendingUp className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-sm">Trend</CardTitle>
            </div>
            <div className="text-xs text-muted-foreground">
              Min: {stats.trend.min.toFixed(2)} | Max:{" "}
              {stats.trend.max.toFixed(2)}
            </div>
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
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
                interval={tickInterval}
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
                stroke="var(--color-trend)"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* CHART 3: SEASONAL */}
      <Card>
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Icons.waves className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-sm">Seasonal</CardTitle>
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
        </CardHeader>
        <CardContent className="pb-4">
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
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
                interval={tickInterval}
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
                stroke="var(--color-seasonal)"
                fill="var(--color-seasonal)"
                fillOpacity={0.2}
                isAnimationActive={false}
              />
            </AreaChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* CHART 4: RESIDUAL */}
      <Card>
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Icons.activity className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-sm">Residual</CardTitle>
            </div>
            <div className="text-xs text-muted-foreground">
              Noise Bound: {stats.residual.min.toFixed(2)} to{" "}
              {stats.residual.max.toFixed(2)}
            </div>
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
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
                interval={tickInterval}
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
                stroke="var(--color-residual)"
                strokeWidth={0}
                dot={{ r: 2, fill: "var(--color-residual)" }}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  );
}

export function DecompositionChart({
  preprocessingId,
}: DecompositionChartProps) {
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

      <Tabs defaultValue={availableParams[0]} className="w-full">
        <TabsList className="mb-4 flex-wrap h-auto gap-2 bg-transparent p-0">
          {availableParams.map((param) => (
            <TabsTrigger
              key={param}
              value={param}
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-lg px-4 py-2 border"
            >
              {getParamLabel(param)}
            </TabsTrigger>
          ))}
        </TabsList>

        {/* PHASE 2: Pass extracted data dynamically to the child view */}
        {availableParams.map((param) => (
          <TabsContent key={param} value={param} className="mt-0">
            <ParameterDecompositionView
              param={param}
              paramData={report.parameters[param]}
              decompositionMethod={report.decomposition_method}
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
