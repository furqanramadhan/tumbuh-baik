// File: d:\tumbuk-baik new\tumbuh-baik\apps\nextjs\src\components\view\calender\lstm-yearly-calendar.tsx
"use client";

import React, { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getLSTMDaily } from "@/lib/fetch/files.fetch";
import clsx from "clsx";
import { format, startOfMonth, getDaysInMonth, getDay } from "date-fns";
import { id } from "date-fns/locale";
import { CalendarDays, ChevronDown, Loader2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";

interface MonthCalendarData {
  month: string;
  year: number;
  monthKey: string;
  firstDayOffset: number;
  daysInMonth: number;
  days: Map<number, "sangatCocok" | "cukupCocok" | "tidakCocok">;
  totalSangatCocok: number;
  totalCukupCocok: number;
  totalTidakCocok: number;
}

interface SuitabilitySummary {
  months: MonthCalendarData[];
  totalSangatCocok: number;
  totalCukupCocok: number;
  totalTidakCocok: number;
}

// Helper function untuk mengambil nilai parameter dengan fallback
const getParameterValue = (parameters: any, paramNames: string[]): number => {
  if (!parameters) return 0;

  for (const name of paramNames) {
    const value = parameters?.[name]?.forecast_value;
    if (value !== undefined && value !== null && !isNaN(value)) {
      return value;
    }
  }
  return 0;
};

const getSuitability = (
  rain: number,
  temp: number,
  humidity: number,
  radiation: number,
) => {
  const criteria = {
    isRainSesuai: rain >= 5.7 && rain <= 16.7,
    isRainTooLow: rain < 5.7,
    isRainTooHigh: rain > 16.7,
    isTempSesuai: temp >= 24 && temp <= 29,
    isHumiditySesuai: humidity >= 33 && humidity <= 90,
    isRadiationSesuai: radiation >= 15 && radiation <= 25,
    isRadiationTooLow: radiation < 15,
    isRadiationTooHigh: radiation > 25,
  };

  const isSuitable =
    criteria.isRainSesuai &&
    criteria.isTempSesuai &&
    criteria.isHumiditySesuai &&
    criteria.isRadiationSesuai;

  if (isSuitable) {
    return { type: "sangatCocok" as const };
  }

  const isPlantableWithWarning =
    !criteria.isRainTooLow &&
    criteria.isTempSesuai &&
    criteria.isHumiditySesuai &&
    !criteria.isRadiationTooLow &&
    (criteria.isRainTooHigh || criteria.isRadiationTooHigh);

  if (isPlantableWithWarning) {
    return { type: "cukupCocok" as const };
  }

  return { type: "tidakCocok" as const };
};

const getSuitabilitySummary = (data: any[]): SuitabilitySummary => {
  if (!data || data.length === 0) {
    return {
      months: [],
      totalSangatCocok: 0,
      totalCukupCocok: 0,
      totalTidakCocok: 0,
    };
  }

  const sortedData = [...data]
    .filter((item) => !item.isPlaceholder)
    .sort(
      (a, b) =>
        new Date(a.forecast_date).getTime() -
        new Date(b.forecast_date).getTime(),
    );

  const monthsMap = new Map<
    string,
    {
      month: string;
      year: number;
      firstDayOffset: number;
      daysInMonth: number;
      days: Map<number, "sangatCocok" | "cukupCocok" | "tidakCocok">;
    }
  >();

  sortedData.forEach((item) => {
    const date = new Date(item.forecast_date);
    const monthKey = format(date, "yyyy-MM");
    const monthName = format(date, "MMMM", { locale: id });
    const year = date.getFullYear();
    const day = date.getDate();

    const rain = getParameterValue(item.parameters, [
      "RR_imputed",
      "RR",
      "PRECTOTCORR",
    ]);
    const temp = getParameterValue(item.parameters, [
      "TAVG",
      "T2M",
      "T2M_MAX",
      "TMAX",
    ]);
    const humidity = getParameterValue(item.parameters, [
      "RH_AVG_preprocessed",
      "RH_AVG",
      "RH2M",
    ]);
    const radiation = getParameterValue(item.parameters, ["ALLSKY_SFC_SW_DWN"]);

    const suitability = getSuitability(rain, temp, humidity, radiation);

    if (!monthsMap.has(monthKey)) {
      const firstDay = startOfMonth(date);
      const firstDayOfWeek = getDay(firstDay);
      const offset = firstDayOfWeek === 0 ? 6 : firstDayOfWeek - 1;

      monthsMap.set(monthKey, {
        month: monthName,
        year: year,
        firstDayOffset: offset,
        daysInMonth: getDaysInMonth(date),
        days: new Map(),
      });
    }

    const monthData = monthsMap.get(monthKey)!;
    monthData.days.set(day, suitability.type);
  });

  const months: MonthCalendarData[] = Array.from(monthsMap.entries()).map(
    ([key, data]) => {
      let totalSangatCocok = 0;
      let totalCukupCocok = 0;
      let totalTidakCocok = 0;

      data.days.forEach((type) => {
        if (type === "sangatCocok") totalSangatCocok++;
        else if (type === "cukupCocok") totalCukupCocok++;
        else totalTidakCocok++;
      });

      return {
        month: data.month,
        year: data.year,
        monthKey: key,
        firstDayOffset: data.firstDayOffset,
        daysInMonth: data.daysInMonth,
        days: data.days,
        totalSangatCocok,
        totalCukupCocok,
        totalTidakCocok,
      };
    },
  );

  return {
    months,
    totalSangatCocok: months.reduce((sum, m) => sum + m.totalSangatCocok, 0),
    totalCukupCocok: months.reduce((sum, m) => sum + m.totalCukupCocok, 0),
    totalTidakCocok: months.reduce((sum, m) => sum + m.totalTidakCocok, 0),
  };
};

const MiniCalendar: React.FC<{ monthData: MonthCalendarData }> = ({
  monthData,
}) => {
  const dayNames = ["S", "S", "R", "K", "J", "S", "M"];

  const calendarGrid: (number | null)[] = [];

  for (let i = 0; i < monthData.firstDayOffset; i++) {
    calendarGrid.push(null);
  }

  for (let day = 1; day <= monthData.daysInMonth; day++) {
    calendarGrid.push(day);
  }

  const weeks: (number | null)[][] = [];
  for (let i = 0; i < calendarGrid.length; i += 7) {
    weeks.push(calendarGrid.slice(i, i + 7));
  }

  if (weeks.length > 0 && weeks[weeks.length - 1].length < 7) {
    const lastWeek = weeks[weeks.length - 1];
    while (lastWeek.length < 7) {
      lastWeek.push(null);
    }
  }

  const getDayColor = (day: number | null) => {
    if (day === null) return "bg-transparent";
    const type = monthData.days.get(day);
    if (!type) return "bg-gray-100 text-gray-400";

    switch (type) {
      case "sangatCocok":
        return "bg-green-400 text-green-900 font-medium";
      case "cukupCocok":
        return "bg-yellow-300 text-yellow-900 font-medium";
      case "tidakCocok":
        return "bg-red-400 text-red-900 font-medium";
      default:
        return "bg-gray-100";
    }
  };

  const isBulanCocokTanam = monthData.totalTidakCocok <= 15;
  const headerBgColor = isBulanCocokTanam
    ? "bg-green-100 border-green-300"
    : "bg-red-100 border-red-300";

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className={clsx("px-2 py-1.5 border-b", headerBgColor)}>
        <div className="flex items-center justify-between">
          <span className="font-semibold text-xs">
            {monthData.month} {monthData.year}
          </span>
          <div className="flex items-center gap-1.5 text-[10px]">
            <span className="flex items-center gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400"></span>
              {monthData.totalSangatCocok}
            </span>
            <span className="flex items-center gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-yellow-300"></span>
              {monthData.totalCukupCocok}
            </span>
            <span className="flex items-center gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span>
              {monthData.totalTidakCocok}
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-7 bg-muted/30">
        {dayNames.map((name, idx) => (
          <div
            key={idx}
            className="text-center text-[10px] font-medium text-muted-foreground py-1"
          >
            {name}
          </div>
        ))}
      </div>

      <div className="p-0.5">
        {weeks.map((week, weekIdx) => (
          <div key={weekIdx} className="grid grid-cols-7 gap-0.5">
            {week.map((day, dayIdx) => (
              <div
                key={dayIdx}
                className={clsx(
                  "aspect-square flex items-center justify-center text-[10px] rounded-sm",
                  getDayColor(day),
                )}
              >
                {day}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export function LSTMYearlyCalendar() {
  const [isOpen, setIsOpen] = useState(false);

  const {
    data: forecastData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["lstm-yearly-calendar"],
    queryFn: async () => {
      const info = await getLSTMDaily(1, 10);
      const total = info.total || 365;
      const all = await getLSTMDaily(1, total);

      const sorted = all.items
        .map((item: any) => ({
          ...item,
          forecast_date: new Date(item.forecast_date).toISOString(),
        }))
        .sort(
          (a: any, b: any) =>
            new Date(a.forecast_date).getTime() -
            new Date(b.forecast_date).getTime(),
        );

      return sorted.slice(0, 365);
    },
  });

  const summary = useMemo(() => {
    if (!forecastData || forecastData.length === 0) return null;
    return getSuitabilitySummary(forecastData);
  }, [forecastData]);

  const yearRange = useMemo(() => {
    if (!summary || summary.months.length === 0) return "";
    const firstMonth = summary.months[0];
    const lastMonth = summary.months[summary.months.length - 1];
    if (firstMonth.year === lastMonth.year) {
      return `${firstMonth.year}`;
    }
    return `${firstMonth.year} - ${lastMonth.year}`;
  }, [summary]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-center py-12 gap-2">
            <Loader2 className="w-5 h-5 animate-spin text-primary" />
            <span className="text-muted-foreground">
              Memuat data kalender tanam...
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !summary) {
    return (
      <Card className="border-destructive">
        <CardContent className="pt-6">
          <Alert variant="destructive">
            <AlertDescription>
              Terjadi kesalahan saat memuat data kalender. Silakan coba lagi.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CalendarDays className="w-5 h-5" />
              <div>
                <CardTitle className="text-base">
                  Kalender Tanam {yearRange}
                </CardTitle>
                <CardDescription className="text-xs">
                  Ringkasan kesesuaian tanam selama {summary.months.length}{" "}
                  bulan
                </CardDescription>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden sm:flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-green-400" />
                  <span className="text-sm font-medium">
                    {summary.totalSangatCocok}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-yellow-300" />
                  <span className="text-sm font-medium">
                    {summary.totalCukupCocok}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-400" />
                  <span className="text-sm font-medium">
                    {summary.totalTidakCocok}
                  </span>
                </div>
              </div>

              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="gap-1">
                  <span className="text-xs">
                    {isOpen ? "Tutup Detail" : "Lihat Detail"}
                  </span>
                  <ChevronDown
                    className={clsx(
                      "h-4 w-4 transition-transform",
                      isOpen && "rotate-180",
                    )}
                  />
                </Button>
              </CollapsibleTrigger>
            </div>
          </div>
        </CardHeader>

        {/* Ringkasan Per Bulan (Tampil saat ditutup) */}
        {!isOpen && (
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
              {summary.months.map((monthData) => {
                const isBulanCocokTanam = monthData.totalTidakCocok <= 15;

                return (
                  <div
                    key={monthData.monthKey}
                    className={clsx(
                      "border-2 rounded-lg p-4 transition-all hover:shadow-md",
                      isBulanCocokTanam
                        ? "border-green-400 bg-green-50/50"
                        : "border-red-400 bg-red-50/50",
                    )}
                  >
                    <div className="space-y-3">
                      <div className="font-bold text-base text-center">
                        {monthData.month}
                      </div>

                      <div className="flex justify-center">
                        <Badge
                          variant={
                            isBulanCocokTanam ? "default" : "destructive"
                          }
                          className="px-3 py-1"
                        >
                          {isBulanCocokTanam
                            ? "✓ Cocok Tanam"
                            : "✕ Tidak Cocok"}
                        </Badge>
                      </div>

                      <div className="text-center text-sm text-muted-foreground">
                        <div className="font-medium">
                          {monthData.totalSangatCocok +
                            monthData.totalCukupCocok}{" "}
                          dari {monthData.daysInMonth} hari
                        </div>
                        <div className="text-xs mt-0.5">dapat tanam</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 pt-3 border-t">
              <p className="text-xs text-center text-muted-foreground">
                Klik <strong>Lihat Detail</strong> untuk melihat kalender
                lengkap per tanggal
              </p>
            </div>
          </CardContent>
        )}

        {/* Detail Kalender (Tampil saat dibuka) */}
        <CollapsibleContent>
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-2">
              {summary.months.map((monthData) => (
                <MiniCalendar key={monthData.monthKey} monthData={monthData} />
              ))}
            </div>

            <div className="flex flex-wrap items-center justify-center gap-4 mt-4 pt-3 border-t text-xs text-muted-foreground">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-sm bg-green-400"></div>
                <span>Sesuai</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-sm bg-yellow-300"></div>
                <span>Waspada, masih dapat tanam</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-sm bg-red-400"></div>
                <span>Tidak sesuai</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 rounded-sm bg-gray-100 border"></div>
                <span>Tidak ada data</span>
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
