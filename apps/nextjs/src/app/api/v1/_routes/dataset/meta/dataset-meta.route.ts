import { Hono } from "hono";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { parseError } from "@/lib/utils";
import mongoose from "mongoose";

interface ConversionInfo {
  originalFormat: string;
  convertedTo: string;
  isMultiFile: boolean;
  filesProcessed: number;
  totalRecords: number;
  fileSize: number;
  processingDetails?: {
    filesSuccessful: number;
    filesFailed: number;
    totalFiles: number;
    duplicatesRemoved: number;
    dateRange: {
      start: string | null;
      end: string | null;
      years?: number;
    };
    failedFiles: string[];
    warnings: string[];
  };
}

interface UploadResponse {
  message: string;
  data: any;
  conversionInfo: ConversionInfo;
}

const datasetMetaRoute = new Hono();

// GET - Ambil semua metadata dataset
datasetMetaRoute.get("/", async (c) => {
  try {
    await db();
    const datasets = await DatasetMeta.find({ deletedAt: null }) // 👈 hanya yg aktif
      .sort({ uploadDate: -1 })
      .lean();
    return c.json({ data: datasets }, 200);
  } catch (error) {
    console.error("Get dataset meta error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// DELETE - Soft delete dataset (move to recycle bin)
datasetMetaRoute.patch("/:collectionName/delete", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    const dataset = await DatasetMeta.findOneAndUpdate(
      { collectionName },
      { $set: { deletedAt: new Date() } }, // 👈 set kolom deletedAt
      { new: true }
    );

    if (!dataset) return c.json({ message: "Dataset not found" }, 404);
    console.log("Updated dataset:", dataset);

    return c.json({ message: "Dataset moved to recycle bin", success: true, data: dataset }, 200);
  } catch (error) {
    console.error("Soft delete dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// Status endpoint with enhanced logic
datasetMetaRoute.get("/:collectionName/status", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    const dataset = await DatasetMeta.findOne({ collectionName }).lean();

    if (!dataset) {
      return c.json({ message: "Dataset not found" }, 404);
    }

    const status = (dataset as any).status as string;

    // Determine what operations are allowed
    const canPreprocess = ["raw", "latest"].includes(status);

    // ENHANCED: Different refresh logic for API vs non-API datasets
    const canRefresh = (dataset as any).isAPI
      ? status !== "archived" // API datasets can refresh unless archived
      : ["raw", "latest", "preprocessed", "validated"].includes(status); // Non-API datasets (manual refresh)

    const canReactivate = status === "archived";

    // Check if cleaned collection exists
    let hasCleanedCollection = false;
    if (status === "preprocessed" || status === "validated") {
      try {
        const cleanedCollectionName = `${collectionName}_cleaned`;
        const CleanedModel = mongoose.models[cleanedCollectionName] || mongoose.model(cleanedCollectionName, new mongoose.Schema({}, { strict: false }), cleanedCollectionName);

        const count = await CleanedModel.countDocuments().limit(1);
        hasCleanedCollection = count > 0;
      } catch {
        hasCleanedCollection = false;
      }
    }

    return c.json({
      collectionName,
      status,
      isAPI: (dataset as any).isAPI,
      canPreprocess,
      canRefresh,
      canReactivate,
      hasCleanedCollection,
      operations: {
        preprocessing: canPreprocess ? "allowed" : `not allowed - status is ${status}`,
        refresh: canRefresh ? ((dataset as any).isAPI ? "allowed via NASA Power refresh" : "manual refresh (to be implemented)") : (dataset as any).isAPI ? `not allowed - status is ${status}` : `not allowed - status is ${status}`,
        reactivation: canReactivate ? "allowed" : "not allowed - not archived",
      },
    });
  } catch (error) {
    console.error("Get dataset status error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

datasetMetaRoute.patch("/:collectionName/reactivate", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    const dataset = await DatasetMeta.findOne({ collectionName });
    if (!dataset) {
      return c.json({ message: "Dataset not found" }, 404);
    }

    if ((dataset as any).status !== "archived") {
      return c.json({ message: "Only archived datasets can be reactivated" }, 400);
    }

    // ENHANCED: Determine new status based on dataset type and data freshness
    let newStatus: string;

    if ((dataset as any).isAPI) {
      // API datasets go to "latest" when reactivated (will be refreshed via API)
      newStatus = "latest";
    } else {
      // Non-API datasets: check data freshness to determine raw vs latest
      try {
        const dynamicModel = mongoose.models[collectionName] || mongoose.model(collectionName, new mongoose.Schema({}, { strict: false }), collectionName);

        // FIXED: Properly type the query result and handle potential array/single document
        const latestRecord = (await dynamicModel.findOne({}).sort({ Date: -1 }).lean()) as any; // Type assertion to any to access Date property

        if (latestRecord && latestRecord.Date) {
          const today = new Date();
          const recordDate = new Date(latestRecord.Date);
          const todayString = today.toISOString().slice(0, 10);
          const recordString = recordDate.toISOString().slice(0, 10);

          if (recordString === todayString) {
            newStatus = "latest"; // Data is current
          } else {
            newStatus = "raw"; // Data needs refresh/preprocessing
          }
        } else {
          newStatus = "raw"; // No date data found, default to raw
        }
      } catch (error) {
        console.warn("Could not check data freshness, defaulting to raw:", error);
        newStatus = "raw";
      }
    }

    const updated = await DatasetMeta.findOneAndUpdate({ collectionName }, { $set: { status: newStatus } }, { new: true });

    return c.json({
      message: "Dataset reactivated successfully",
      data: updated,
      statusTransition: {
        from: "archived",
        to: newStatus,
        reason: (dataset as any).isAPI ? "API dataset reactivated to latest for refresh" : "Non-API dataset reactivated based on data freshness",
      },
    });
  } catch (error: any) {
    console.error("Reactivate dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

datasetMetaRoute.post("/:collectionName/refresh", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    const dataset = await DatasetMeta.findOne({ collectionName }).lean();

    if (!dataset) {
      return c.json({ message: "Dataset not found" }, 404);
    }

    // Only allow refresh for non-API datasets
    if ((dataset as any).isAPI) {
      return c.json(
        {
          message: "API datasets should use NASA Power refresh endpoints",
          suggestedEndpoint: "/api/v1/nasa-power/refresh/:id",
        },
        400
      );
    }

    // Validate status can be refreshed
    const currentStatus = (dataset as any).status as string;
    const refreshableStatuses = ["raw", "latest", "preprocessed", "validated"];

    if (!refreshableStatuses.includes(currentStatus)) {
      return c.json(
        {
          message: `Cannot refresh dataset with status '${currentStatus}'`,
          refreshableStatuses,
        },
        400
      );
    }

    // Check if dataset was preprocessed/validated (has _cleaned collection)
    const wasPreprocessed = currentStatus === "preprocessed" || currentStatus === "validated";

    if (wasPreprocessed) {
      // Delete the _cleaned collection if it exists
      const cleanedCollectionName = `${collectionName}_cleaned`;
      try {
        const CleanedModel = mongoose.models[cleanedCollectionName] || mongoose.model(cleanedCollectionName, new mongoose.Schema({}, { strict: false }), cleanedCollectionName);
        await CleanedModel.collection.drop();
        console.log(`✅ Deleted cleaned collection: ${cleanedCollectionName} (manual refresh)`);
      } catch (error: any) {
        if (error.code !== 26) {
          // 26 = NamespaceNotFound
          console.warn(`⚠️ Warning deleting cleaned collection ${cleanedCollectionName}:`, error.message);
        }
      }
    }

    // PLACEHOLDER: For future manual refresh implementation
    return c.json(
      {
        message: "Manual refresh for user-uploaded datasets will be implemented later",
        currentStatus: currentStatus,
        collectionName: collectionName,
        wasPreprocessed,
        cleanedCollectionDeleted: wasPreprocessed,
        note: "This endpoint is reserved for future manual refresh functionality",
        implementation: {
          suggestion: "Users can re-upload newer data files to refresh non-API datasets",
          expectedFlow: "Upload new file → Smart status detection → Replace data",
        },
      },
      501
    ); // Not Implemented
  } catch (error: any) {
    console.error("Non-API refresh error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// PATCH - Restore soft deleted dataset from recycle bin
datasetMetaRoute.patch("/:collectionName/restore", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    const dataset = await DatasetMeta.findOneAndUpdate(
      { collectionName },
      { deletedAt: null },
      { new: true },
    );

    if (!dataset) return c.json({ message: "Dataset not found" }, 404);

    return c.json(
      { message: "Dataset restored successfully", data: dataset },
      200,
    );
  } catch (error) {
    console.error("Restore dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// GET - Display soft deleted datasets in recycle bin
datasetMetaRoute.get("/recycle-bin", async (c) => {
  try {
    await db();
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;

    const total = await DatasetMeta.countDocuments({
      deletedAt: { $ne: null },
    });
    console.log("Total deleted items:", total); // 👈 ad
    const datasets = await DatasetMeta.find({ deletedAt: { $ne: null } })
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ deletedAt: -1 })
      .lean();

    console.log("Found datasets:", datasets.length);

    return c.json({
      message: "Recycle bin data retrieved successfully",
      data: {
        items: datasets,
        total,
        currentPage: page,
        totalPages: Math.ceil(total / pageSize),
        pageSize,
      },
    });
  } catch (error) {
    console.error("Get recycle bin error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// GET - Buat slug untuk setiap dataset baru
datasetMetaRoute.get("/:slug", async (c) => {
  try {
    await db();
    const { slug } = c.req.param();
    console.log("[DEBUG] API dataset-meta called with slug:", slug);
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;
    const sortBy = c.req.query("sortBy") || "Date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    const meta = await DatasetMeta.findOne({ collectionName: slug }).lean();
    if (!meta) return c.json({ message: "Dataset not found" }, 404);

    let Model;
    try {
      Model = mongoose.model(slug);
    } catch {
      Model = (mongoose.models[slug] || mongoose.model(slug, new mongoose.Schema({}, { strict: false }), slug)) as mongoose.Model<any>;
    }

    const sortQuery: Record<string, 1 | -1> = {
      [sortBy]: sortOrder === "desc" ? -1 : 1,
    };
    const totalData = await Model.countDocuments();

    const data = await Model.find()
      .sort(sortQuery)
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .lean();

    return c.json(
      {
        message: "Success",
        data: {
          meta,
          items: data,
          total: totalData,
          currentPage: page,
          totalPages: Math.ceil(totalData / pageSize),
          pageSize,
          sortBy,
          sortOrder,
        },
      },
      200
    );
  } catch (error) {
    console.error("Error fetching dynamic dataset:", error);
    return c.json({ message: "Server error" }, 500);
  }
});

// Route untuk fetch semua data tanpa pagination (khusus chart)
datasetMetaRoute.get("/:slug/chart-data", async (c) => {
  try {
    await db();
    const { slug } = c.req.param();
    console.log("[DEBUG] API chart-data called with slug:", slug);

    // Gunakan type assertion untuk meta
    const meta = (await DatasetMeta.findOne({
      collectionName: slug,
    }).lean()) as {
      name: string;
      collectionName: string;
      columns: string[];
      [key: string]: any;
    } | null;

    if (!meta) return c.json({ message: "Dataset not found" }, 404);

    // Validasi kolom Date
    const hasDateColumn = meta.columns.some((col: string) => col.toLowerCase() === "date");

    if (!hasDateColumn) {
      return c.json(
        {
          message: "Dataset tidak memiliki kolom Date",
          data: null,
        },
        200
      );
    }

    let Model;
    try {
      Model = mongoose.model(slug);
    } catch {
      Model = (mongoose.models[slug] || mongoose.model(slug, new mongoose.Schema({}, { strict: false }), slug)) as mongoose.Model<any>;
    }

    // Fetch SEMUA data, sort by Date ascending untuk chart
    const allData = await Model.find().sort({ Date: 1 }).lean();

    // Filter hanya kolom numerik (exclude Date dan kolom string)
    const firstItem = allData[0] || {};
    const numericColumns = meta.columns.filter((col: string) => {
      if (col.toLowerCase() === "date") return false;
      const value = firstItem[col];
      return typeof value === "number" || !isNaN(Number(value));
    });

    return c.json(
      {
        message: "Success",
        data: {
          items: allData,
          numericColumns,
          dateColumn: "Date",
        },
      },
      200
    );
  } catch (error) {
    console.error("Error fetching chart data:", error);
    return c.json({ message: "Server error" }, 500);
  }
});

// PUT - Update dataset meta by ID or collectionName
datasetMetaRoute.patch("/:idOrSlug", async (c) => {
  try {
    await db();

    const { idOrSlug } = c.req.param();

    // Get request body (existing logic remains the same)
    let body: Record<string, any> = {};

    try {
      const contentType = c.req.header("content-type") || "";

      if (contentType.includes("application/json")) {
        body = (await c.req.json()) as Record<string, any>;
      } else {
        const queries = c.req.queries();
        Object.keys(queries).forEach((key) => {
          body[key] = queries[key][0];
        });
      }
    } catch {
      const queries = c.req.queries();
      Object.keys(queries).forEach((key) => {
        body[key] = queries[key][0];
      });
    }

    if (Object.keys(body).length === 0) {
      return c.json(
        {
          message: "No data provided for update. Send data as JSON body or query parameters.",
        },
        400
      );
    }

    // ADDED: Status transition validation
    if (body.status) {
      const allowedStatuses = ["raw", "latest", "preprocessed", "validated", "archived"];

      if (!allowedStatuses.includes(body.status)) {
        return c.json({ message: `Invalid status: ${body.status}` }, 400);
      }

      // Get current dataset to check current status
      let currentDataset;
      if (mongoose.Types.ObjectId.isValid(idOrSlug)) {
        currentDataset = await DatasetMeta.findById(idOrSlug).lean();
      } else {
        currentDataset = await DatasetMeta.findOne({
          collectionName: idOrSlug,
        }).lean();
      }

      if (!currentDataset) {
        return c.json({ message: "Dataset not found" }, 404);
      }

      // ADDED: Validate status transitions according to lifecycle
      const currentStatus = (currentDataset as any).status;
      const newStatus = body.status;

      const validTransitions: Record<string, string[]> = {
        raw: ["latest", "preprocessed", "archived"],
        latest: ["raw", "preprocessed", "archived"],
        preprocessed: ["raw", "validated", "archived"],
        validated: ["raw", "archived"],
        archived: ["raw", "latest"], // Allow reactivation
      };

      if (!validTransitions[currentStatus]?.includes(newStatus)) {
        return c.json(
          {
            message: `Invalid status transition: ${currentStatus} → ${newStatus}. Valid transitions from ${currentStatus}: ${validTransitions[currentStatus]?.join(", ") || "none"}`,
            currentStatus,
            attemptedStatus: newStatus,
            validTransitions: validTransitions[currentStatus] || [],
          },
          400
        );
      }

      console.log(`✅ Status transition validated: ${currentStatus} → ${newStatus}`);
    }

    let updatedDataset;

    // Check if idOrSlug is a valid MongoDB ObjectId
    if (mongoose.Types.ObjectId.isValid(idOrSlug)) {
      // Update by ID
      updatedDataset = await DatasetMeta.findByIdAndUpdate(idOrSlug, { $set: body }, { new: true, runValidators: true, lean: true });
    } else {
      // Update by collectionName (slug)
      updatedDataset = await DatasetMeta.findOneAndUpdate({ collectionName: idOrSlug }, { $set: body }, { new: true, runValidators: true, lean: true });
    }

    if (!updatedDataset) {
      return c.json({ message: "Dataset not found" }, 404);
    }

    return c.json(
      {
        message: "Dataset metadata updated successfully",
        data: updatedDataset,
      },
      200
    );
  } catch (error) {
    console.error("Update dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// POST - Upload dataset metadata + records
datasetMetaRoute.post("/", async (c) => {
  try {
    await db();

    const contentType = c.req.header("content-type") || "";

    // Handle XLSX via multipart/form-data
    if (contentType.includes("multipart/form-data")) {
      return await handleXlsxUpload(c);
    }

    // Handle CSV/JSON via JSON body
    const body = await c.req.json();

    const requiredFields = ["name", "source", "fileType", "data"];
    for (const field of requiredFields) {
      if (!body[field]) {
        return c.json({ message: `${field} is required` }, 400);
      }
    }

    if (!["csv", "json"].includes(body.fileType)) {
      return c.json({ message: "fileType must be 'csv' or 'json'" }, 400);
    }

    const {
      name,
      source,
      fileType,
      data,
      filename = `${name}.${fileType}`,
      description = "",
      status: requestedStatus, // Don't use default here
      collectionName: rawCollectionName,
    } = body;

    if (!Array.isArray(data) || data.length === 0) {
      return c.json({ message: "data must be a non-empty array" }, 400);
    }

    // ADDED: Smart status detection based on data freshness
    let finalStatus = requestedStatus || "raw"; // Default to raw

    if (!requestedStatus) {
      // Auto-detect status based on data freshness
      const today = new Date();
      const todayString = today.toISOString().slice(0, 10); // YYYY-MM-DD

      // Find the newest date in the dataset
      let newestDate: Date | null = null;

      for (const item of data) {
        if (item.Date) {
          const itemDate = new Date(item.Date);
          if (!newestDate || itemDate > newestDate) {
            newestDate = itemDate;
          }
        }
      }

      if (newestDate) {
        const newestDateString = newestDate.toISOString().slice(0, 10);
        console.log(`Newest date in dataset: ${newestDateString}, Today: ${todayString}`);

        // If newest date matches today, set status to "latest"
        if (newestDateString === todayString) {
          finalStatus = "latest";
          console.log(`✅ Dataset contains today's data, setting status to 'latest'`);
        } else if (newestDate < today) {
          // Data is older than today, keep as "raw"
          const daysDiff = Math.floor((today.getTime() - newestDate.getTime()) / (1000 * 60 * 60 * 24));
          finalStatus = "raw";
          console.log(`Dataset is ${daysDiff} days behind, setting status to 'raw'`);
        } else {
          // Future date (shouldn't happen but handle gracefully)
          finalStatus = "raw";
          console.log(`Dataset contains future dates, setting status to 'raw'`);
        }
      } else {
        console.log("No Date column found in dataset, defaulting to 'raw'");
        finalStatus = "raw";
      }
    }

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB

    // FIXED: Collection name should be original, not _cleaned
    const collectionName =
      rawCollectionName?.trim() ||
      name
        .trim()
        .replace(/[^a-zA-Z0-9\s]/g, "")
        .replace(/\s+/g, " ");

    const fileSize = Buffer.byteLength(JSON.stringify(data));

    if (fileSize > MAX_FILE_SIZE) {
      return c.json({ message: "File size exceeds 16MB limit" }, 400);
    }

    const totalRecords = data.length || 0;
    const columns = data[0] ? Object.keys(data[0]) : [];

    // Insert data to dynamic collection
    const parsedData = data.map((item) => ({
      ...item,
      Date: item.Date ? new Date(item.Date) : null,
    }));

    let dynamicModel;
    try {
      dynamicModel = mongoose.model(collectionName);
    } catch {
      dynamicModel = mongoose.model(collectionName, new mongoose.Schema({}, { strict: false }), collectionName);
    }

    await dynamicModel.insertMany(parsedData);

    // FIXED: Save metadata pointing to original collection
    const newDataset = await DatasetMeta.create({
      name: name.trim(),
      source: source.trim(),
      filename,
      collectionName, // FIXED: Points to original collection, not _cleaned
      fileType,
      status: finalStatus, // Use smart-detected status
      description,
      fileSize,
      totalRecords,
      columns,
      isAPI: false, // Always false for manual uploads
    });

    return c.json(
      {
        message: "Dataset uploaded and metadata saved successfully",
        data: newDataset,
        statusInfo: {
          detectedStatus: finalStatus,
          reason: finalStatus === "latest" ? "Dataset contains today's data" : "Dataset requires refresh or preprocessing",
          wasAutoDetected: !requestedStatus,
        },
      },
      201
    );
  } catch (error) {
    console.error("Upload dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// DELETE - Hapus dataset (metadata + collection)
datasetMetaRoute.delete("/:collectionName", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    // Decode the collection name (handle URL encoding)
    const decodedCollectionName = decodeURIComponent(collectionName);

    // 1. Get metadata
    const metadata = await DatasetMeta.findOne({
      collectionName: decodedCollectionName,
    }).lean();

    if (!metadata) {
      return c.json({ message: "Dataset not found" }, 404);
    }

    // 2. Determine collections to delete
    const collectionsToDelete: string[] = [];
    const metaStatus = (metadata as any).status as string;

    // Always delete the original collection
    collectionsToDelete.push(decodedCollectionName);

    // If dataset is preprocessed/validated, also delete cleaned collection
    if (
      (metaStatus === "preprocessed" || metaStatus === "validated") &&
      !decodedCollectionName.endsWith("_cleaned")
    ) {
      const cleanedCollectionName = `${decodedCollectionName}_cleaned`;
      collectionsToDelete.push(cleanedCollectionName);
      console.log(
        `Dataset is ${metaStatus}, will also delete cleaned collection: ${cleanedCollectionName}`
      );
    }

    // 3. Delete metadata first
    await DatasetMeta.deleteOne({ collectionName: decodedCollectionName });

    // 4. Drop all identified collections
    const deletedCollections: string[] = [];
    const failedDeletions: string[] = [];

    const mongoDb = mongoose.connection.db;
    if (!mongoDb) throw new Error("MongoDB connection not ready");

    for (const colName of collectionsToDelete) {
      try {
        const collections = await mongoDb
          .listCollections({ name: colName })
          .toArray();

        if (collections.length > 0) {
          await mongoDb.dropCollection(colName);
          deletedCollections.push(colName);
          console.log(`✅ Dropped collection: ${colName}`);
        } else {
          console.log(`ℹ️ Collection ${colName} not found, skipping`);
        }
      } catch (dropError: any) {
        console.error(`⚠️ Error dropping collection ${colName}:`, dropError);
        failedDeletions.push(colName);
      }
    }

    return c.json(
      {
        message: "Dataset deleted successfully",
        deletedCollections,
        failedDeletions: failedDeletions.length > 0 ? failedDeletions : undefined,
        originalStatus: metaStatus,
      },
      200
    );
  } catch (error) {
    console.error("Delete dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

async function handleXlsxUpload(c: any) {
  try {
    const formData = await c.req.formData();

    // Extract files from formData
    const name = (formData.get("name") as string)?.trim();
    const source = (formData.get("source") as string)?.trim();
    const description = ((formData.get("description") as string) || "").trim();
    const requestedStatus = ((formData.get("status") as string) || "").trim();
    const isMultiFile = formData.get("isMultiFile") === "true";
    // Basic validation
    if (!name || !source) {
      return c.json({ message: "name and source are required" }, 400);
    }

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    const MAX_TOTAL_SIZE = 100 * 1024 * 1024; // 100 MB for multi-file
    const MAX_FILES = 50; // Maximum 50 files

    let conversionResult;
    let filename: string;
    let filesProcessed = 0;

    // ✅ Handle Multi-file XLSX merge
    if (isMultiFile) {
      console.log("Processing multi-file XLSX upload...");

      // Collect all files from formData
      const files: Array<{ buffer: number[]; filename: string }> = [];
      let totalSize = 0;

      // Get all file entries (should be named "files")
      const fileEntries = formData.getAll("files");

      if (!fileEntries || fileEntries.length === 0) {
        return c.json({ message: "No files provided for multi-file upload" }, 400);
      }

      if (fileEntries.length > MAX_FILES) {
        return c.json(
          {
            message: `Too many files. Maximum ${MAX_FILES} files allowed per batch`,
          },
          400
        );
      }

      // Process each file
      for (const entry of fileEntries) {
        if (entry instanceof File) {
          const file = entry as File;

          // Validate file extension
          if (!file.name.toLowerCase().endsWith(".xlsx")) {
            return c.json(
              {
                message: `File ${file.name} is not a .xlsx file`,
              },
              400
            );
          }

          // Validate individual file size
          if (file.size > MAX_FILE_SIZE) {
            return c.json(
              {
                message: `File ${file.name} exceeds 16MB limit`,
              },
              400
            );
          }

          totalSize += file.size;

          // Validate total size
          if (totalSize > MAX_TOTAL_SIZE) {
            return c.json(
              {
                message: `Total file size exceeds 100MB limit`,
              },
              400
            );
          }

          const arrayBuffer = await file.arrayBuffer();
          const fileBuffer = Array.from(new Uint8Array(arrayBuffer));

          files.push({
            buffer: fileBuffer,
            filename: file.name,
          });
        }
      }

      if (files.length === 0) {
        return c.json(
          {
            message: "No valid .xlsx files found",
          },
          400
        );
      }

      filesProcessed = files.length;
      filename = `${name.replace(/\s+/g, "_")}_merged_${files.length}_files.csv`;

      // ✅ Send to Flask for merging
      const flaskFormData = new FormData();

      // Add each file to FormData for Flask
      files.forEach((fileData) => {
        const buffer = Buffer.from(fileData.buffer);
        const flaskFile = new File([buffer], fileData.filename, {
          type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        });
        flaskFormData.append("files", flaskFile);
      });

      try {
        console.log(`Sending ${files.length} files to Flask merge service...`);
        const response = await fetch("http://localhost:5001/api/v1/convert/xlsx-merge-csv", {
          method: "POST",
          body: flaskFormData,
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Flask merge service error: ${response.status} ${response.statusText}. Response: ${errorText}`);
        }

        const responseText = await response.text();
        try {
          conversionResult = JSON.parse(responseText);
        } catch (jsonError) {
          console.error("JSON Parse Error:", jsonError);
          console.error("Raw Response:", responseText.substring(0, 500) + "...");
          throw new Error(`Invalid JSON response from merge service: ${jsonError}`);
        }
      } catch (fetchError) {
        console.error("Flask merge service error:", fetchError);
        return c.json({ message: `Failed to connect to merge service: ${fetchError}` }, 500);
      }
    }
    // ✅ Handle Single XLSX file
    else {
      console.log("Processing single XLSX file upload...");

      const file = formData.get("file") as File;

      if (!file || !(file instanceof File)) {
        return c.json({ message: "file is required for single file upload" }, 400);
      }

      // Validate file extension
      if (!file.name.toLowerCase().endsWith(".xlsx")) {
        return c.json({ message: "Only .xlsx files are allowed" }, 400);
      }

      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        return c.json({ message: "File size exceeds 16MB limit" }, 400);
      }

      filesProcessed = 1;
      filename = file.name.replace(".xlsx", ".csv");

      // Convert File to buffer array
      const arrayBuffer = await file.arrayBuffer();
      const fileBuffer = Array.from(new Uint8Array(arrayBuffer));

      // ✅ Send to Flask for single file conversion
      const flaskFormData = new FormData();
      const buffer = Buffer.from(fileBuffer);
      const flaskFile = new File([buffer], file.name, {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      });
      flaskFormData.append("file", flaskFile);

      try {
        console.log(`Sending single file to Flask conversion service...`);
        const response = await fetch("http://localhost:5001/api/v1/convert/xlsx-to-csv", {
          method: "POST",
          body: flaskFormData,
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Flask service error: ${response.status} ${response.statusText}. Response: ${errorText}`);
        }

        const responseText = await response.text();
        try {
          conversionResult = JSON.parse(responseText);
        } catch (jsonError) {
          console.error("JSON Parse Error:", jsonError);
          console.error("Raw Response:", responseText.substring(0, 500) + "...");
          throw new Error(`Invalid JSON response from conversion service: ${jsonError}`);
        }
      } catch (fetchError) {
        console.error("Flask service error:", fetchError);
        return c.json({ message: `Failed to connect to conversion service: ${fetchError}` }, 500);
      }
    }

    // ✅ Check conversion result (same for both single and multi)
    if (conversionResult.status !== "success") {
      return c.json(
        {
          message: `XLSX conversion failed: ${conversionResult.error || "Unknown error"}`,
        },
        400
      );
    }

    // Extract converted data
    const processedData = conversionResult.records;
    const totalRecords = conversionResult.record_count;
    const columns = conversionResult.columns;

    if (!Array.isArray(processedData) || processedData.length === 0) {
      return c.json({ message: "Conversion resulted in empty data" }, 400);
    }

    // Status detection for XLSX uploads
    let finalStatus = requestedStatus || "raw";

    if (!requestedStatus) {
      // Auto-detect status based on data freshness
      const today = new Date();
      const todayString = today.toISOString().slice(0, 10); // YYYY-MM-DD

      // Find the newest date in the processed data
      let newestDate: Date | null = null;

      for (const item of processedData) {
        if (item.Date) {
          const itemDate = new Date(item.Date);
          if (!newestDate || itemDate > newestDate) {
            newestDate = itemDate;
          }
        }
      }

      if (newestDate) {
        const newestDateString = newestDate.toISOString().slice(0, 10);
        console.log(`XLSX - Newest date: ${newestDateString}, Today: ${todayString}`);

        // If newest date matches today, set status to "latest"
        if (newestDateString === todayString) {
          finalStatus = "latest";
          console.log(`✅ XLSX dataset contains today's data, setting status to 'latest'`);
        } else if (newestDate < today) {
          const daysDiff = Math.floor((today.getTime() - newestDate.getTime()) / (1000 * 60 * 60 * 24));
          finalStatus = "raw";
          console.log(`XLSX dataset is ${daysDiff} days behind, setting status to 'raw'`);
        } else {
          finalStatus = "raw";
          console.log(`XLSX dataset contains future dates, setting status to 'raw'`);
        }
      } else {
        console.log("No Date column found in XLSX dataset, defaulting to 'raw'");
        finalStatus = "raw";
      }
    }

    // Generate collection name
    const collectionName = name
      .trim()
      .replace(/[^a-zA-Z0-9\s]/g, "")
      .replace(/\s+/g, " ");

    const fileSize = Buffer.byteLength(JSON.stringify(processedData));

    // Validate processed data size
    if (fileSize > MAX_TOTAL_SIZE) {
      return c.json({ message: "Processed data exceeds size limit" }, 400);
    }

    // Parse data with Date conversion
    const parsedData = processedData.map((item: any) => ({
      ...item,
      Date: item.Date ? new Date(item.Date) : null,
    }));

    // Create or get dynamic model
    let dynamicModel;
    try {
      dynamicModel = mongoose.model(collectionName);
    } catch {
      dynamicModel = mongoose.model(collectionName, new mongoose.Schema({}, { strict: false }), collectionName);
    }

    // Insert to MongoDB
    await dynamicModel.insertMany(parsedData);

    // Save metadata
    const newDataset = await DatasetMeta.create({
      name: name.trim(),
      source: source.trim(),
      filename,
      collectionName,
      fileType: "csv", // Always saved as CSV after conversion
      status: finalStatus,
      description: description.trim(),
      fileSize,
      totalRecords,
      columns,
      isAPI: false,
    });

    // ✅ Return response with interface UploadResponse
    const response: UploadResponse = {
      message: "Dataset uploaded and metadata saved successfully",
      data: newDataset,
      conversionInfo: {
        originalFormat: "xlsx",
        convertedTo: "csv",
        isMultiFile,
        filesProcessed,
        totalRecords,
        fileSize,
      },
    };

    // Add additional info for multi-file uploads
    if (isMultiFile && conversionResult.processing_summary) {
      response.conversionInfo.processingDetails = {
        filesSuccessful: conversionResult.processing_summary.files_processed,
        filesFailed: conversionResult.processing_summary.files_failed,
        totalFiles: conversionResult.processing_summary.total_files,
        duplicatesRemoved: conversionResult.processing_summary.duplicates_removed,
        dateRange: conversionResult.processing_summary.date_range,
        failedFiles: conversionResult.failed_files || [],
        warnings: conversionResult.warnings || [],
      };
    }

    // Status information in response
    (response as any).statusInfo = {
      detectedStatus: finalStatus,
      reason: finalStatus === "latest" ? "XLSX dataset contains today's data" : "XLSX dataset requires refresh or preprocessing",
      wasAutoDetected: !requestedStatus,
    };

    console.log(`✅ XLSX conversion completed: ${isMultiFile ? "Multi-file" : "Single"} upload successful with status: ${finalStatus}`);
    return c.json(response, 201);
  } catch (error) {
    console.error("XLSX upload error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
}
export default datasetMetaRoute;
