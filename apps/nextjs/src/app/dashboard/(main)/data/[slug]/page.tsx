import { GetDatasetBySlug } from "@/lib/fetch/files.fetch";
import { notFound } from "next/navigation";
// import { Metadata } from "next";
import ChartSection from "@/app/dashboard/_components/chart/dataset-chart";
import DynamicMainTable from "../_components/dynamic-table";
// import { DecompositionChart } from "@/app/dashboard/_components/chart/decomposition-chart";

interface Props {
  params: Promise<{ slug: string }>;
}

export default async function DatasetDetailPage({ params }: Props) {
  const { slug } = await params;
  console.log("[DEBUG] slug from route:", slug);
  const result = await GetDatasetBySlug(slug).catch(() => null);
  if (!result) return notFound();

  const { meta } = result;

  return (
    <div className="p-5">
      <h2 className="text-xl font-semibold mb-4">{meta.name}</h2>
      <DynamicMainTable
        collectionName={meta.collectionName}
        columns={meta.columns}
        datasetId={meta._id}
        datasetName={meta.name}
        isAPI={meta.isAPI || false}
      />
      <ChartSection collectionName={meta.collectionName} />

      {/* Show decomposition chart only for preprocessed datasets */}
      {/* {meta.status === "preprocessed" && (
        <DecompositionChart collectionName={meta.collectionName} />
      )} */}
    </div>
  );
}
