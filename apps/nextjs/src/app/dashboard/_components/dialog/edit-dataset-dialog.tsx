"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import React, { useState } from "react";
import { useEffect } from "react";
import {
  UpdateDatasetMeta,
  getDatasetRefreshStatus,
  refreshNasaPowerDataset,
} from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTrigger,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Icons } from "@/app/dashboard/_components/icons";
import { ConfirmationUpdateModal } from "@/components/ui/modal/confirmation-update-modal";
import toast from "react-hot-toast";

interface EditDatasetDialogProps {
  dataset: {
    _id: string;
    name: string;
    source: string;
    collectionName: string;
    description?: string;
    status: string;
    isAPI?: boolean;
    apiConfig?: {
      type: string;
      params?: any;
    };
    lastUpdated?: string;
  };
  children: React.ReactNode;
}

export default function EditDatasetDialog({
  dataset,
  children,
}: EditDatasetDialogProps) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [isUpdateConfirmOpen, setIsUpdateConfirmOpen] = useState(false);
  const [refreshStatus, setRefreshStatus] = useState({
    canRefresh: false,
    daysSinceLastRecord: 0,
    lastRecordDate: "",
    message: "",
    isLoading: false,
  });

  // Form state - initialize with dataset values
  const [form, setForm] = useState({
    name: dataset?.name || "",
    source: dataset?.source || "",
    collectionName: dataset?.collectionName || "",
    description: dataset?.description || "",
    status: dataset?.status || "raw",
  });

  // Reset form when dialog opens or dataset changes
  useEffect(() => {
    if (open && dataset) {
      setForm({
        name: dataset.name || "",
        source: dataset.source || "",
        collectionName: dataset.collectionName || "",
        description: dataset.description || "",
        status: dataset.status || "raw",
      });
    }
  }, [open, dataset]);

  // useEffect for NASA latest date
  useEffect(() => {
    if (open && dataset.isAPI && dataset.apiConfig?.type === "nasa-power") {
      setRefreshStatus((prev) => ({ ...prev, isLoading: true }));

      getDatasetRefreshStatus(dataset._id, true)
        .then((status) => {
          setRefreshStatus({
            canRefresh: status.canRefresh,
            daysSinceLastRecord: status.daysSinceLastRecord || 0,
            lastRecordDate: status.lastRecordDate || "",
            message: status.message,
            isLoading: false,
          });
        })
        .catch((error) => {
          console.error("Error fetching refresh status:", error);
          setRefreshStatus({
            canRefresh: true,
            daysSinceLastRecord: 0,
            lastRecordDate: "",
            message: "Unable to check status",
            isLoading: false,
          });
        });
    }
  }, [open, dataset._id, dataset.isAPI, dataset.apiConfig?.type]);

  // Mutation update - FIXED VERSION
  const { mutate: updateDataset, isPending: isPending } = useMutation({
    mutationKey: ["update-dataset", dataset._id],
    mutationFn: async (data: typeof form) => {
      if (!dataset?._id) {
        console.error("❌ Dataset ID is missing!");
        throw new Error("Dataset ID is missing");
      }

      // Validate required fields
      if (!data.name?.trim()) {
        throw new Error("Nama dataset tidak boleh kosong");
      }
      if (!data.source?.trim()) {
        throw new Error("Sumber data tidak boleh kosong");
      }

      // Build update payload - send all editable fields
      const updatePayload: {
        name: string;
        source: string;
        collectionName: string;
        description: string;
        status?: string;
      } = {
        name: data.name.trim(),
        source: data.source.trim(),
        collectionName: data.collectionName.trim(),
        description: data.description?.trim() || "",
      };

      // Include status if it changed
      if (data.status && data.status !== dataset.status) {
        updatePayload.status = data.status;
      }

      console.log("📤 Update Dataset Request:");
      console.log("   Dataset ID:", dataset._id);
      console.log("   Payload:", updatePayload);

      try {
        const result = await UpdateDatasetMeta(dataset._id, updatePayload);
        console.log("✅ Update successful, result:", result);
        return result;
      } catch (error: any) {
        console.error("❌ UpdateDatasetMeta error:", error);
        console.error("   Response:", error?.response?.data);
        console.error("   Status:", error?.response?.status);
        throw error;
      }
    },
    onSuccess: (result) => {
      console.log("✅ Update mutation succeeded:", result);
      
      toast.success("Dataset berhasil diperbarui", {
        duration: 3000,
        position: "bottom-right",
      });

      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.invalidateQueries({ queryKey: ["dataset", dataset._id] });

      // Close modals
      setOpen(false);
      setIsUpdateConfirmOpen(false);
    },
    onError: (error: any) => {
      console.error("❌ Update mutation failed");
      console.error("   Error object:", error);
      console.error("   Response data:", error?.response?.data);
      console.error("   Status code:", error?.response?.status);

      // Extract error message with multiple fallbacks
      let errorMessage = "Gagal memperbarui dataset";

      if (error?.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error?.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error?.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error?.message) {
        errorMessage = error.message;
      }

      // Add HTTP status code for debugging
      if (error?.response?.status) {
        errorMessage = `[${error.response.status}] ${errorMessage}`;
      }

      toast.error(errorMessage, {
        duration: 5000,
        position: "bottom-right",
      });

      setIsUpdateConfirmOpen(false);
    },
  });

  // Mutation refresh (for NASA POWER datasets)
  const { mutate: refreshDataset, isPending: isRefreshing } = useMutation({
    mutationKey: ["refresh-nasa-dataset", dataset._id],
    mutationFn: () => {
      if (!dataset?._id) {
        throw new Error("Dataset ID is missing");
      }
      return refreshNasaPowerDataset(dataset._id);
    },
    onSuccess: (data) => {
      // Check if there were records updated
      if (
        data.data?.newRecordsCount === 0 ||
        data.message?.includes("up to date") ||
        data.message?.includes("No new data")
      ) {
        toast(
          `Dataset sudah memiliki data terbaru\nData terakhir: ${new Date(
            data.data?.lastUpdated || dataset.lastUpdated || "",
          ).toLocaleDateString("id-ID")}`,
          {
            duration: 5000,
            icon: "ℹ️",
            position: "bottom-right",
          },
        );
      } else {
        toast.success(
          `Berhasil memperbarui ${
            data.data?.newRecordsCount || 0
          } data baru\nTotal data: ${
            data.data?.dataset?.totalRecords || "N/A"
          }`,
          {
            duration: 5000,
            position: "bottom-right",
          },
        );
      }

      // Update refresh status after successful refresh
      setRefreshStatus({
        canRefresh: false,
        daysSinceLastRecord: 0,
        lastRecordDate: new Date().toISOString(),
        message: "Dataset sudah up-to-date",
        isLoading: false,
      });

      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
    onError: (error: any) => {
      console.error("❌ Refresh failed, error:", error);

      // Special handling for "already up to date" messages
      if (
        error?.response?.data?.message?.includes("up to date") ||
        error?.message?.includes("up to date")
      ) {
        toast(
          `Dataset sudah memiliki data terbaru\nData terakhir: ${new Date(
            refreshStatus.lastRecordDate || dataset.lastUpdated || "",
          ).toLocaleDateString("id-ID")}`,
          {
            duration: 5000,
            icon: "ℹ️",
            position: "bottom-right",
          },
        );

        setRefreshStatus({
          canRefresh: false,
          daysSinceLastRecord: 0,
          lastRecordDate: dataset.lastUpdated || "",
          message: "Dataset sudah up-to-date",
          isLoading: false,
        });
        return;
      }

      const errorMessage =
        error?.response?.data?.message || "Gagal memperbarui dataset";
      toast.error(errorMessage);
    },
  });

  // Function handle refresh
  const handleRefreshClick = () => {
    if (refreshStatus.isLoading) {
      toast("Memeriksa status data...", {
        icon: "ℹ️",
        position: "bottom-right",
      });
      return;
    }

    if (!refreshStatus.canRefresh) {
      toast(
        `Dataset sudah memiliki data terbaru\nData terakhir: ${new Date(
          refreshStatus.lastRecordDate || dataset.lastUpdated || "",
        ).toLocaleDateString("id-ID")}`,
        {
          duration: 5000,
          icon: "ℹ️",
          position: "bottom-right",
        },
      );
      return;
    }

    refreshDataset();
  };

  const handleSubmitClick = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate form before showing confirmation
    if (!form.name.trim()) {
      toast.error("Nama dataset tidak boleh kosong");
      return;
    }
    if (!form.source.trim()) {
      toast.error("Sumber data tidak boleh kosong");
      return;
    }

    console.log("✅ Form validation passed, opening confirmation modal");
    setIsUpdateConfirmOpen(true);
  };

  const handleConfirmUpdate = () => {
    console.log("✅ Update confirmed, calling mutation with form:", form);
    updateDataset(form);
  };

  return (
    <>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>{children}</DialogTrigger>
        <DialogContent className="w-[95vw] max-w-[500px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Dataset</DialogTitle>
            <DialogDescription>Ubah metadata dataset.</DialogDescription>
          </DialogHeader>

          {/* Main form */}
          <form onSubmit={handleSubmitClick} className="space-y-4">
            <div className="grid gap-2">
              <Label htmlFor="name">
                Nama Dataset <span className="text-red-500">*</span>
              </Label>
              <Input
                id="name"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="Masukkan nama dataset"
                required
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="source">
                Sumber <span className="text-red-500">*</span>
              </Label>
              <select
                id="source"
                value={form.source}
                onChange={(e) => setForm({ ...form, source: e.target.value })}
                className="border rounded px-3 py-2"
                required
              >
                <option value="">Pilih sumber data...</option>
                <option value="Data BMKG (https://dataonline.bmkg.go.id/)">
                  Data BMKG (https://dataonline.bmkg.go.id/)
                </option>
                <option value="Data NASA (https://power.larc.nasa.gov/)">
                  Data NASA (https://power.larc.nasa.gov/)
                </option>
              </select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="collectionName">Nama Koleksi (Opsional)</Label>
              <Input
                id="collectionName"
                value={form.collectionName}
                onChange={(e) =>
                  setForm({ ...form, collectionName: e.target.value })
                }
                placeholder="Nama koleksi di database"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="status">Status</Label>
              <select
                id="status"
                value={form.status}
                onChange={(e) => setForm({ ...form, status: e.target.value })}
                className="border rounded px-3 py-2"
              >
                {dataset.isAPI && dataset.apiConfig?.type === "nasa-power" ? (
                  <>
                    <option value="raw">Raw</option>
                    <option value="latest">Latest</option>
                    <option value="preprocessed">Preprocessed</option>
                    <option value="validated">Validated</option>
                    <option value="archived">Archived</option>
                  </>
                ) : (
                  <>
                    <option value="raw">Raw</option>
                    <option value="cleaned">Cleaned</option>
                    <option value="validated">Validated</option>
                    <option value="archived">Archived</option>
                  </>
                )}
              </select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="description">Deskripsi</Label>
              <Input
                id="description"
                value={form.description}
                onChange={(e) =>
                  setForm({ ...form, description: e.target.value })
                }
                placeholder="Deskripsi dataset (opsional)"
              />
            </div>

            {/* Action buttons */}
            <div className="space-y-4 pt-4 border-t">
              {/* NASA POWER Refresh Section */}
              {dataset.isAPI && dataset.apiConfig?.type === "nasa-power" && (
                <div className="flex flex-col gap-2 p-3 bg-gray-50 rounded-lg border">
                  <div className="text-sm">
                    <p className="font-medium text-gray-700 mb-2">
                      Status Data NASA POWER
                    </p>
                    {refreshStatus.isLoading ? (
                      <p className="text-xs text-blue-600 flex items-center gap-1">
                        <Icons.refresh className="h-3 w-3 animate-spin" />
                        Memeriksa status...
                      </p>
                    ) : (
                      <p className="text-xs">
                        {refreshStatus.canRefresh ? (
                          <span className="text-green-600 font-medium">
                            ✓ Tersedia {refreshStatus.daysSinceLastRecord} hari
                            data baru
                          </span>
                        ) : (
                          <span className="text-gray-500">
                            ✓ Up-to-date:{" "}
                            {new Date(
                              dataset.lastUpdated || "",
                            ).toLocaleDateString("id-ID", {
                              day: "numeric",
                              month: "short",
                              year: "numeric",
                            })}
                          </span>
                        )}
                      </p>
                    )}
                  </div>

                  <Button
                    type="button"
                    variant={refreshStatus.canRefresh ? "secondary" : "outline"}
                    onClick={handleRefreshClick}
                    disabled={
                      isRefreshing ||
                      !refreshStatus.canRefresh ||
                      refreshStatus.isLoading
                    }
                    className="flex items-center justify-center gap-2 w-full"
                    size="sm"
                  >
                    <Icons.refresh
                      className={`h-4 w-4 ${
                        isRefreshing ? "animate-spin" : ""
                      }`}
                    />
                    {isRefreshing
                      ? "Memperbarui data..."
                      : refreshStatus.canRefresh
                        ? `Refresh Data (${refreshStatus.daysSinceLastRecord} hari)`
                        : "Data Terbaru"}
                  </Button>
                </div>
              )}

              {/* Bottom Action Buttons */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
                <div className="flex items-center gap-2 w-full sm:w-auto">
                  <DialogClose asChild>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="flex-1 sm:flex-none"
                    >
                      Batal
                    </Button>
                  </DialogClose>
                  <Button
                    type="submit"
                    disabled={isPending}
                    className="flex items-center justify-center gap-2 flex-1 sm:flex-none"
                    size="sm"
                  >
                    <Icons.save className="h-4 w-4" />
                    {isPending ? "Menyimpan..." : "Simpan"}
                  </Button>
                </div>
              </div>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      {/* Update Confirmation Modal */}
      <ConfirmationUpdateModal
        isOpen={isUpdateConfirmOpen}
        setIsOpen={setIsUpdateConfirmOpen}
        onConfirm={handleConfirmUpdate}
        datasetName={form.name}
        isUpdating={isPending}
      />
    </>
  );
}
