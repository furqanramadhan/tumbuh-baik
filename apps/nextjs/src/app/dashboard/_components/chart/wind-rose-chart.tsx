"use client";

import { useState } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  SPEED_GRADIENTS,
  SPEED_BINS,
  SPEED_LABELS,
  createCumulativeStacks,
  processWindRoseData,
} from "@/lib/wind-rose-utils";

interface WindRoseChartProps {
  data: any[];
  directionColumn: string;
  speedColumn: string;
  title?: string;
  domainMax?: number | "auto";
}

// Custom tooltip function di-define di luar balutan tag JSX Tooltip content
const CustomTooltip = ({ active, payload, showPercentage }: any) => {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0].payload;

  return (
    <div className="bg-background border rounded-lg shadow-sm p-3 text-sm min-w-48">
      <div className="font-semibold mb-2 pb-2 border-b">{data.direction}</div>
      {SPEED_BINS.map((bin) => {
        if (data[bin] === 0) return null;
        return (
          <div
            key={bin}
            className="flex justify-between items-center gap-6 py-0.5"
          >
            <span className="text-muted-foreground text-xs">
              {SPEED_LABELS[bin]} ({bin} m/s)
            </span>
            <span className="font-medium">
              {showPercentage ? `${data[bin].toFixed(1)}%` : `${data[bin]}x`}
            </span>
          </div>
        );
      })}
      <div className="flex justify-between gap-6 pt-2 mt-2 border-t font-semibold">
        <span className="text-muted-foreground">Total</span>
        <span>{showPercentage ? `${data.total.toFixed(1)}%` : data.total}</span>
      </div>
    </div>
  );
};

export default function WindRoseChart({
  data,
  directionColumn,
  speedColumn,
  title,
  domainMax = "auto",
}: WindRoseChartProps) {
  const [showPercentage, setShowPercentage] = useState(false);

  const { roseData, calmPercentage, totalValid } = processWindRoseData(
    data,
    directionColumn,
    speedColumn,
    showPercentage,
  );

  if (totalValid === 0) {
    return (
      <div className="w-full h-[400px] flex flex-col items-center justify-center border rounded-lg bg-muted/10">
        {title && <div className="font-semibold mb-4">{title}</div>}
        <div className="text-center space-y-2 text-sm text-muted-foreground">
          Data tidak cukup untuk Wind Rose
        </div>
      </div>
    );
  }

  const stackedData = createCumulativeStacks(roseData);

  return (
    <div className="w-full space-y-4 border rounded-lg p-4 bg-muted/5">
      {title && <h3 className="font-semibold text-center text-lg">{title}</h3>}
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-muted/50 rounded-full text-xs">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
          <span className="font-medium">
            Calm: {calmPercentage.toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            id="percentage-mode"
            checked={showPercentage}
            onCheckedChange={setShowPercentage}
          />
          <Label htmlFor="percentage-mode" className="text-xs cursor-pointer">
            Persentase
          </Label>
        </div>
      </div>

      {/* Tetapkan batas tinggi absolute 400px agar seragam dengan line chart */}
      <div className="w-full h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={stackedData}>
            <defs>
              <linearGradient id="color_1" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={SPEED_GRADIENTS["0.5-2.1"].start}
                  stopOpacity={0.9}
                />
                <stop
                  offset="95%"
                  stopColor={SPEED_GRADIENTS["0.5-2.1"].end}
                  stopOpacity={0.9}
                />
              </linearGradient>
              <linearGradient id="color_2" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={SPEED_GRADIENTS["2.1-3.6"].start}
                  stopOpacity={0.9}
                />
                <stop
                  offset="95%"
                  stopColor={SPEED_GRADIENTS["2.1-3.6"].end}
                  stopOpacity={0.9}
                />
              </linearGradient>
              <linearGradient id="color_3" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={SPEED_GRADIENTS["3.6-5.7"].start}
                  stopOpacity={0.9}
                />
                <stop
                  offset="95%"
                  stopColor={SPEED_GRADIENTS["3.6-5.7"].end}
                  stopOpacity={0.9}
                />
              </linearGradient>
              <linearGradient id="color_4" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={SPEED_GRADIENTS["5.7-8.8"].start}
                  stopOpacity={0.9}
                />
                <stop
                  offset="95%"
                  stopColor={SPEED_GRADIENTS["5.7-8.8"].end}
                  stopOpacity={0.9}
                />
              </linearGradient>
              <linearGradient id="color_5" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={SPEED_GRADIENTS[">8.8"].start}
                  stopOpacity={0.9}
                />
                <stop
                  offset="95%"
                  stopColor={SPEED_GRADIENTS[">8.8"].end}
                  stopOpacity={0.9}
                />
              </linearGradient>
            </defs>

            <PolarGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <PolarAngleAxis
              dataKey="direction"
              tick={{ fill: "hsl(var(--foreground))", fontSize: 12 }}
            />
            {/* Apply dynamic domain */}
            <PolarRadiusAxis
              angle={30}
              domain={[0, domainMax]}
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
            />

            {/* Syntax pemanggilan Tooltip diubah agar menghindari error 'Element type is invalid' */}
            <Tooltip
              content={(props) => (
                <CustomTooltip {...props} showPercentage={showPercentage} />
              )}
            />

            <Radar
              name=">8.8"
              dataKey="stack_5"
              stroke="none"
              fill="url(#color_5)"
              fillOpacity={1}
            />
            <Radar
              name="5.7-8.8"
              dataKey="stack_4"
              stroke="none"
              fill="url(#color_4)"
              fillOpacity={1}
            />
            <Radar
              name="3.6-5.7"
              dataKey="stack_3"
              stroke="none"
              fill="url(#color_3)"
              fillOpacity={1}
            />
            <Radar
              name="2.1-3.6"
              dataKey="stack_2"
              stroke="none"
              fill="url(#color_2)"
              fillOpacity={1}
            />
            <Radar
              name="0.5-2.1"
              dataKey="stack_1"
              stroke="none"
              fill="url(#color_1)"
              fillOpacity={1}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex flex-wrap justify-center gap-3 mt-4">
        {SPEED_BINS.map((speed) => (
          <div
            key={speed}
            className="flex items-center gap-1.5"
            title={SPEED_LABELS[speed]}
          >
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: SPEED_GRADIENTS[speed].start }}
            />
            <span className="text-xs font-medium">{speed} m/s</span>
          </div>
        ))}
      </div>

      {/* Reference Addendum */}
      <div className="text-center mt-2 pb-2 text-[10px] text-muted-foreground/60 italic">
        * Klasifikasi kecepatan berdasarkan Skala Beaufort (Simbolon et. al,
        2022)
      </div>
    </div>
  );
}
