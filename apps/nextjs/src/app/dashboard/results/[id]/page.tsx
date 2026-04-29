"use client";
import { useQuery } from "@tanstack/react-query";
import {
  getPreprocessingReportById,
  getDecompositionByPreprocessingId,
} from "@/lib/fetch/files.fetch";
import { Icons } from "@/app/dashboard/_components/icons";
import Link from "next/link";
import { format } from "date-fns";
import { use } from "react";
import { Button } from "@/components/ui/button";
import { id as localeId } from "date-fns/locale";
import { PreprocessingSummary } from "../../_components/preprrocessing-summary/preprocessing-summary";
import { MetricsCharts } from "../../_components/chart/metrics-charts";
import ComparisonChart from "@/app/dashboard/_components/chart/comparison-chart";
import { DecompositionChart } from "@/app/dashboard/_components/chart/decomposition-chart";

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function ResultDetailsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  // 1. Fetch preprocessing Report
  const {
    data: preprocessingReport,
    isLoading: isLoadingReport,
    error: errorReport,
  } = useQuery({
    queryKey: ["preprocessing-report", id],
    queryFn: () => getPreprocessingReportById(id),
  });

  // 2. Fetch decomposition report
  const { data: decompositionReport, isLoading: isLoadingDecomposition } =
    useQuery({
      queryKey: ["decomposition-report", id],
      queryFn: () => getDecompositionByPreprocessingId(id),
      // Only run if we actually have an ID to avoid premature failing
      enabled: !!id,
    });

  const isLoading = isLoadingReport || isLoadingDecomposition;
  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <Icons.spinner className="text-muted-foreground h-8 w-8 animate-spin" />
      </div>
    );
  }
  if (errorReport || !preprocessingReport) {
    return (
      <div className="p-6">
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">
              Error Loading Report
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>Could not find preprocessing results for ID: {id}</p>
            <Button asChild className="mt-4" variant="outline">
              <Link href="/dashboard/results">Back to Results</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }
  // Parse the preprocessing timestamp
  const dateObj =
    typeof preprocessingReport.preprocessing_timestamp === "object"
      ? new Date((preprocessingReport.preprocessing_timestamp as any).$date)
      : new Date(preprocessingReport.preprocessing_timestamp);

  return (
    <div className="space-y-6 p-6">
      {/* --- BREADCRUMBS --- */}
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbLink href="/dashboard/data">Data</BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbLink href="/dashboard/results">Results</BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage className="font-mono text-xs">{id}</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      {/* --- HEADER --- */}
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            {preprocessingReport.cleaned_collection_name.replace(
              "_cleaned",
              "",
            )}
          </h1>
          <p className="text-muted-foreground flex items-center gap-2 mt-1">
            <span className="uppercase font-semibold tracking-wider text-xs bg-secondary px-2 py-1 rounded">
              {preprocessingReport.dataset_type} Data
            </span>
            <span>•</span>
            <span>
              Cleaned on{" "}
              {format(dateObj, "dd MMMM yyyy, HH:mm", { locale: localeId })}
            </span>
          </p>
        </div>
      </div>

      {/* --- CONTENT PLACEHOLDERS (For Next Phases) --- */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Original Records
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {preprocessingReport.record_count.original.toLocaleString()}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Processed Records
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {preprocessingReport.record_count.processed.toLocaleString()}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize text-green-600">
              {preprocessingReport.status}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Decomposition Parameters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {decompositionReport
                ? Object.keys(decompositionReport.parameters).length
                : 0}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="mt-8 rounded-lg border-2 border-dashed border-muted p-8 text-center text-muted-foreground">
        <PreprocessingSummary report={preprocessingReport} />
        <MetricsCharts report={preprocessingReport} />
      </div>
      <ComparisonChart
        originalCollectionName={preprocessingReport.original_collection_name}
        cleanedCollectionName={preprocessingReport.cleaned_collection_name}
      />
      {/* PHASE 5: Mount Decomposition Plot */}
      {decompositionReport ? (
        <div className="mt-8">
          {/* Use the page 'id' because the backend API route is /preprocessing-report/:id/decomposition */}
          <DecompositionChart preprocessingId={id} />
        </div>
      ) : (
        <div className="rounded-lg border-2 border-dashed border-muted p-8 text-center text-muted-foreground mt-8">
          <p>Data dekomposisi tidak tersedia untuk dataset ini.</p>
        </div>
      )}
    </div>
  );
}
