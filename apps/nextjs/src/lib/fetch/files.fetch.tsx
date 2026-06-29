import axios from "axios";
import { FLASK_API_URL } from "../env";

export interface DatasetMetaType {
  _id: string;
  name: string;
  source: string;
  collectionName: string;
  description?: string;
  status: string;
  uploadDate: string;
  deletedAt?: string;
  isAPI: boolean; // Added API flag for fetching NASA
  lastUpdated?: string; // Dated latest updating
  apiConfig?: Record<string, any>; // API config
  refreshInfo: {
    canRefresh: boolean;
    daysSinceLastRecord: number;
    lastRecordDate: string;
    startDate: string;
    endDate: string;
  };
}

export interface ValidationMetric {
  gcv_score: number | null;
  trend_preservation_pct: number | null;
  quality_status: string | null;
}

export interface PreprocessingReport {
  _id: { $oid: string } | string;
  dataset_type: "nasa" | "bmkg";
  original_collection_name: string;
  cleaned_collection_name: string;
  preprocessing_timestamp: { $date: string } | string; // Support both formats
  preprocessing_summary: Record<string, any>;
  quality_metrics: Record<string, any>;
  imputation_validation?: Record<string, ValidationMetric>; // for BMKG
  smoothing_validation?: Record<string, ValidationMetric>; // for NASA
  model_coverage: Record<string, any>;
  warnings: string[];
  status: string;
  record_count: {
    original: number;
    processed: number;
  };
}

export interface DecompositionDataPoint {
  Date: string | { $date: string };
  original: number;
  trend: number;
  seasonal: number;
  residual: number;
}

export interface ParameterDecomposition {
  seasonal_strength: number | null;
  metadata?: Record<string, any>; // Handled for NASA specific metadata
  data: DecompositionDataPoint[];
}

export interface DecompositionReport {
  _id: { $oid: string } | string;
  preprocessing_id: { $oid: string } | string;
  dataset_name: string;
  dataset_type: "nasa" | "bmkg";
  decomposition_method: string;
  timestamp: { $date: string } | string;
  parameters: Record<string, ParameterDecomposition>;
}

export interface NasaPowerParams {
  start: string; // format YYYYMMDD
  end: string;
  latitude: number;
  longitude: number;
  parameters: string[];
  community?: string;
}

export interface RefreshResult {
  id: string;
  name: string;
  status: string;
  refreshResult: string;
  newRecordsCount?: number;
  lastRecord?: string;
  reason?: string;
}

export interface RefreshAllResponse {
  message: string;
  data: {
    total: number;
    refreshed: number;
    alreadyUpToDate: number;
    failed: number;
    details: RefreshResult[];
  };
}

const getBaseUrl = () => {
  if (typeof window !== "undefined") {
    return "";
  }

  if (process.env.NODE_ENV === "production") {
    return "https://zonapetik.site";
  }

  // Local development
  return "http://localhost:3000";
};
export async function exportDatasetCsv(
  collectionName: string,
  sortBy = "Date",
  sortOrder = "desc",
) {
  try {
    const response = await axios.get("/api/v1/export-csv/dataset-meta", {
      params: { category: collectionName, sortBy, sortOrder }, // category → collectionName
      responseType: "blob",
    });

    if (response.status === 200) {
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const contentDisposition = response.headers["content-disposition"];
      let filename = `${collectionName}_data_${new Date().toISOString().split("T")[0]}.csv`;

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      link.href = url;
      link.download = filename;
      link.style.display = "none";

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true, message: "File downloaded successfully" };
    }
  } catch (error) {
    console.error("Error exporting CSV:", error);
    return { success: false, message: "Export failed" };
  }
}
export async function exportHoltWinterCsv(
  sortBy = "forecast_date",
  sortOrder = "desc",
) {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(`${baseUrl}/api/v1/export-csv/hw-daily`, {
      params: { sortBy, sortOrder },
      responseType: "blob",
    });

    if (response.status === 200) {
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const filename = `holt_winter_daily_${
        new Date().toISOString().split("T")[0]
      }.csv`;
      link.href = url;
      link.download = filename;
      link.style.display = "none";

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true, message: "File downloaded successfully" };
    }
  } catch (error: any) {
    console.error(
      "Export HW CSV failed:",
      error?.response?.data || error.message,
    );
    return { success: false, message: "Export failed" };
  }
}
export async function exportLSTMForecastCsv(
  sortBy = "forecast_date",
  sortOrder = "desc",
) {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(
      `${baseUrl}/api/v1/export-csv/lstm/daily`,
      {
        params: { sortBy, sortOrder },
        responseType: "blob",
      },
    );

    if (response.status === 200) {
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const filename = `lstm_daily_${new Date().toISOString().split("T")[0]}.csv`;
      link.href = url;
      link.download = filename;
      link.style.display = "none";

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true, message: "File downloaded successfully" };
    }
  } catch (error) {
    console.error("Export LSTM Forecast CSV failed:", error);
    return { success: false, message: "Export failed" };
  }
}
export const getBmkgLive = async () => {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(`${baseUrl}/api/v1/bmkg-live/all`);
    return response.data;
  } catch (error: any) {
    console.error("Error fetching BMKG live data:", error);
    throw new Error(
      error?.response?.data?.description || "Failed to fetch BMKG live data",
    );
  }
};
//bmkg api
export const getBmkgApi = async () => {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(`${baseUrl}/api/v1/bmkg-api/all`);
    return response.data;
  } catch (error: any) {
    console.error("Error fetching BMKG API data:", error);
    throw new Error(
      error?.response?.data?.description || "Failed to fetch BMKG data",
    );
  }
};
// holt winter table
export const getHoltWinterDaily = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/hw/daily", {
      params: { page, pageSize },
    });
    if (res.status === 200) {
      console.log("🟢 Retrieved documents:", res.data.length);
      console.log("✅ HW API response:", res.data.data);
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
        }
      );
    }
  } catch (error) {
    console.error("Error fetching HW Daily:", error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
    };
  }
};
export const getLSTMDaily = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/lstm/daily", {
      params: { page, pageSize },
    });
    if (res.status === 200) {
      console.log("🟢 Retrieved documents:", res.data.length);
      console.log("✅ LSTM API response:", res.data.data);
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
        }
      );
    }
  } catch (error) {
    console.error("Error fetching LSTM Daily:", error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
    };
  }
};
export const getSeeds = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/seeds", {
      params: { page, pageSize },
    });

    if (res.status === 200) {
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
        }
      );
    }
  } catch (error) {
    console.error("Get seeds error:", error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
    };
  }
};
export const createSeed = async (data: { name: string; duration: number }) => {
  try {
    const res = await axios.post("/api/v1/seeds", data);
    return res.data.data;
  } catch (error) {
    console.error("Create seed error:", error);
    throw error;
  }
};
export const updateSeed = async (
  id: string,
  data: { name: string; duration: number },
) => {
  try {
    const res = await axios.put(`/api/v1/seeds/${id}`, data);
    return res.data.data;
  } catch (error) {
    console.error("Update seed error:", error);
    throw error;
  }
};

export const deleteSeed = async (id: string) => {
  try {
    const res = await axios.delete(`/api/v1/seeds/${id}`);
    return res.data;
  } catch (error) {
    console.error("Delete seed error:", error);
    throw error;
  }
};
export const getUsers = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/user", {
      params: { page, pageSize },
    });

    if (res.status === 200) {
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
        }
      );
    }
  } catch (error) {
    console.error("Get users error:", error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
    };
  }
};
export const updateUserRole = async (
  userId: string,
  role: "user" | "admin",
) => {
  try {
    const res = await axios.put(`/api/v1/user/${userId}/role`, { role });
    return res.data.data;
  } catch (error) {
    console.error("Update user role error:", error);
    throw error;
  }
};
export const getAllKuesionerPetani = async () => {
  try {
    const res = await axios.get("/api/v1/kuesioner/petani");
    console.log("getAllKuesionerPetani", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching kuesioner petani:", error);
    throw error;
  }
};

// export const getKuesionerPetaniById = async (id: string) => {
//   try {
//     const res = await axios.get(`/api/v1/kuesioner/petani/${id}`);
//     return res.data;
//   } catch (error) {
//     console.error("Error fetching kuesioner petani by ID:", error);
//     throw error;
//   }
// };
export const getAllKuesionerManajemen = async () => {
  try {
    const res = await axios.get("/api/v1/kuesioner-manajemen/manajemen");
    console.log("getAllKuesionerManajemen", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching kuesioner manajemen:", error);
    throw error;
  }
};
export const getAllKuesionerPeriode = async () => {
  try {
    const res = await axios.get("/api/v1/kuesioner-periode/periode");
    console.log("getAllKuesionerPeriode", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching kuesioner periode:", error);
    throw error;
  }
};
export const GetAllDatasetMeta = async (): Promise<DatasetMetaType[]> => {
  try {
    const res = await axios.get("/api/v1/dataset-meta");
    return res.data.data;
  } catch (error) {
    console.error("Get all dataset meta error:", error);
    throw error;
  }
};
export const GetRecycleBinDatasets = async (page = 1, pageSize = 10) => {
  try {
    console.log("[GetRecycleBinDatasets] Fetching recycle bin datasets", {
      page,
      pageSize,
    });

    const res = await axios.get("/api/v1/dataset-meta/recycle-bin", {
      params: { page, pageSize },
    });

    console.log("[GetRecycleBinDatasets] Response status:", res.status);
    console.log("[GetRecycleBinDatasets] Response data:", res.data);

    if (res.status === 200) {
      const result = res.data.data || {
        items: [],
        total: 0,
        currentPage: 1,
        totalPages: 1,
        pageSize,
      };

      console.log("[GetRecycleBinDatasets] Final result:", result);
      return result;
    }
  } catch (error) {
    console.error("Get recycle bin error:", error);
    throw error;
  }
};
export const SoftDeleteDataset = async (collectionName: string) => {
  try {
    const baseUrl = getBaseUrl();
    const res = await axios.patch(
      `${baseUrl}/api/v1/dataset-meta/${collectionName}/delete`,
    );
    return res.data.data;
  } catch (error) {
    console.error("Soft delete dataset error", error);
    throw error;
  }
};

export const RestoreDataset = async (collectionName: string) => {
  try {
    const baseUrl = getBaseUrl();
    const res = await axios.patch(
      `${baseUrl}/api/v1/dataset-meta/${collectionName}/restore`,
    );
    return res.data.data;
  } catch (error) {
    console.error("Restore dataset error:", error);
    throw error;
  }
};
export const PermanentDeleteDataset = async (collectionName: string) => {
  try {
    const baseUrl = getBaseUrl();
    const res = await axios.delete(
      `${baseUrl}/api/v1/dataset-meta/${collectionName}`,
    );
    return res.data;
  } catch (error) {
    console.error("Permanent delete dataset error:", error);
    throw error;
  }
};

// for dataset table
// lib/fetch/files.fetch.tsx
export async function getDynamicDatasetData(
  slug: string,
  page = 1,
  pageSize = 10,
  sortBy = "Date",
  sortOrder: "asc" | "desc" = "desc",
) {
  try {
    const res = await axios.get(`/api/v1/dataset-meta/${slug}`, {
      params: { page, pageSize, sortBy, sortOrder },
    });

    if (res.status === 200) {
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
          sortBy,
          sortOrder,
        }
      );
    }
  } catch (error) {
    console.error(`❌ Error fetching dataset ${slug}:`, error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
      sortBy,
      sortOrder,
    };
  }
}
export const AddDatasetMeta = async (data: {
  name: string;
  source: string;
  fileType: "csv" | "json";
  filename?: string;
  collectionName?: string;
  status?: string;
  description?: string;
  records: Record<string, any>[]; // hasil parsing CSV atau JSON
}) => {
  try {
    const res = await axios.post("/api/v1/dataset-meta", {
      name: data.name,
      source: data.source,
      fileType: data.fileType,
      filename: data.filename || `${data.name}.${data.fileType}`,
      collectionName: data.collectionName,
      description: data.description || "",
      status: data.status || "raw",
      data: data.records,
    });
    return res.data.data;
  } catch (error) {
    console.error("Add dataset meta error:", error);
    throw error;
  }
};

// forecast config :
export const getForecastConfigs = async () => {
  const response = await axios.get("/api/v1/forecast-config");
  return response.data.data;
};

export const createForecastConfig = async (data: {
  name: string;
  columns: { collectionName: string; columnName: string }[];
  startDate: string; // Format: "2025-01-15"
}) => {
  const response = await axios.post("/api/v1/forecast-config", data);
  return response.data;
};

export const updateForecastConfig = async (
  id: string,
  data: {
    name: string;
    columns: { collectionName: string; columnName: string }[];
    startDate: string;
  },
) => {
  const response = await axios.put(`/api/v1/forecast-config/${id}`, data);
  return response.data;
};

export const deleteForecastConfig = async (id: string) => {
  const response = await axios.delete(`/api/v1/forecast-config/${id}`);
  return response.data;
};

export const getLSTMConfigs = async () => {
  const response = await axios.get("/api/v1/lstm-config");
  return response.data.data;
};

export const createLSTMConfig = async (data: {
  name: string;
  columns: { collectionName: string; columnName: string }[];
  startDate: string;
}) => {
  const response = await axios.post("/api/v1/lstm-config", data);
  return response.data;
};

export const updateLSTMConfig = async (
  id: string,
  data: {
    name: string;
    columns: { collectionName: string; columnName: string }[];
    startDate: string;
  },
) => {
  const response = await axios.put(`/api/v1/lstm-config/${id}`, data);
  return response.data;
};

export const deleteLSTMConfig = async (id: string) => {
  const response = await axios.delete(`/api/v1/lstm-config/${id}`);
  return response.data;
};
export const triggerForecastRun = async () => {
  try {
    // const res = await axios.post("http://localhost:5001/run-forecast");
    const res = await axios.post(`${FLASK_API_URL}/run-forecast`);
    return res.data;
  } catch (error: any) {
    if (error.response?.status === 404) {
      return { message: "Tidak ada config pending" };
    }
    throw error;
  }
};

export const triggerLSTMForecast = async () => {
  try {
    const res = await axios.post(`${FLASK_API_URL}/run-lstm`);
    return res.data;
  } catch (error) {
    console.error("Trigger LSTM forecast failed:", error);
    throw error;
  }
};

// ==================== FARM API FUNCTIONS ====================

// Get all farms (optional filter by userId)
export const getAllFarms = async (userId?: string) => {
  try {
    const params = userId ? { userId } : {};
    const res = await axios.get("/api/v1/farm", { params });
    console.log("getAllFarms", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching all farms:", error);
    throw error;
  }
};

// Get farm by ID
export const getFarmById = async (id: string) => {
  try {
    const res = await axios.get(`/api/v1/farm/${id}`);
    console.log("getFarmById", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching farm by ID:", error);
    throw error;
  }
};

// Get all farms by user ID
export const getFarmsByUserId = async (userId: string) => {
  try {
    const res = await axios.get(`/api/v1/farm/user/${userId}`);
    console.log("getFarmsByUserId", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching farms by user ID:", error);
    throw error;
  }
};

// Create new farm
export const createFarm = async (data: any) => {
  try {
    const res = await axios.post("/api/v1/farm", data);
    console.log("createFarm", res.data);
    return res.data;
  } catch (error) {
    console.error("Create farm error:", error);
    throw error;
  }
};

// Update farm by ID
export const updateFarm = async (id: string, data: any) => {
  try {
    const res = await axios.put(`/api/v1/farm/${id}`, data);
    console.log("updateFarm", res.data);
    return res.data;
  } catch (error) {
    console.error("Update farm error:", error);
    throw error;
  }
};

// Delete farm by ID
export const deleteFarm = async (id: string) => {
  try {
    const res = await axios.delete(`/api/v1/farm/${id}`);
    console.log("deleteFarm", res.data);
    return res.data;
  } catch (error) {
    console.error("Delete farm error:", error);
    throw error;
  }
};
// ==================== DECOMPOSE LSTM API FUNCTIONS ====================

export const getDecomposeLSTM = async () => {
  try {
    const res = await axios.get("/api/v1/decompose-lstm/all");
    console.log("✅ Decompose LSTM API response:", res.data);
    return res.data.data || [];
  } catch (error) {
    console.error("Error fetching Decompose LSTM:", error);
    return [];
  }
};

export const getDecomposeLSTMByDate = async (date: string) => {
  try {
    const res = await axios.get(`/api/v1/decompose-lstm/date/${date}`);
    return res.data.data;
  } catch (error) {
    console.error("Error fetching Decompose LSTM by date:", error);
    throw error;
  }
};

export const getDecomposeLSTMByConfigId = async (configId: string) => {
  try {
    const res = await axios.get(`/api/v1/decompose-lstm/config/${configId}`);
    return res.data.data || [];
  } catch (error) {
    console.error("Error fetching Decompose LSTM by config:", error);
    return [];
  }
};
// Fetch historical data from collection for comparison with forecast
export const fetchHistoricalData = async (
  collectionName: string,
  columnName: string,
) => {
  try {
    const countResponse = await axios.get(
      `/api/v1/dataset-meta/${collectionName}`,
      {
        params: {
          page: 1,
          pageSize: 10,
          sortBy: "Date",
          sortOrder: "asc",
        },
      },
    );

    const total = countResponse.data?.data?.total || 0;
    if (total === 0) return [];

    const response = await axios.get(`/api/v1/dataset-meta/${collectionName}`, {
      params: {
        page: 1,
        pageSize: total,
        sortBy: "Date",
        sortOrder: "asc",
      },
    });

    const items = response.data?.data?.items || [];
    return items
      .filter((item: any) => item.Date && item[columnName] != null)
      .map((item: any) => ({
        date: item.Date,
        value: item[columnName],
      }));
  } catch (error) {
    console.error(
      `Error fetching historical data for ${collectionName}:`,
      error,
    );
    return [];
  }
};
//for dataset detail page
export const GetDatasetBySlug = async (
  slug: string,
): Promise<{ meta: any; items: any[] }> => {
  const baseUrl = getBaseUrl();
  const url = `${baseUrl}/api/v1/dataset-meta/${slug}`;

  const res = await fetch(url, {
    cache: "no-store",
  });

  if (!res.ok) throw new Error("Failed to fetch dataset");

  const json = await res.json();
  return json.data;
};

export const GetChartDataBySlug = async (
  slug: string,
): Promise<{
  items: any[];
  numericColumns: string[];
  dateColumn: string;
} | null> => {
  const baseUrl = getBaseUrl(); // Gunakan helper yang sama
  const url = `${baseUrl}/api/v1/dataset-meta/${slug}/chart-data`;

  const res = await fetch(url, {
    cache: "no-store",
  });

  if (!res.ok) throw new Error("Failed to fetch chart data");

  const json = await res.json();
  return json.data;
};

export const GetComparisonData = async (
  originalCollectionName: string,
  cleanedCollectionName: string,
) => {
  try {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/api/v1/dataset-meta/comparison/${encodeURIComponent(originalCollectionName)}/${encodeURIComponent(cleanedCollectionName)}`;
    const res = await axios.get(url);
    return res.data.data;
  } catch (error) {
    console.error("Get comparison data error:", error);
    throw error;
  }
};

// Check if BMKG dataset can be preprocessed
export const checkBmkgDatasetForPreprocessing = async (
  collectionName: string,
) => {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(
      `${baseUrl}/api/v1/preprocess/bmkg/${collectionName}/validate`,
    );
    return response.data;
  } catch (error) {
    console.error("Error validating BMKG dataset:", error);
    throw error;
  }
};

// Archive dataset function
export const archiveDataset = async (idOrSlug: string) => {
  try {
    // First check current status to provide better feedback
    let currentDataset;
    try {
      if (idOrSlug.match(/^[0-9a-fA-F]{24}$/)) {
        // It's an ObjectId, get by ID
        const allDatasets = await GetAllDatasetMeta();
        currentDataset = allDatasets.find((d) => d._id === idOrSlug);
      } else {
        // It's a collection name, get status info
        const statusInfo = await getDatasetStatusInfo(idOrSlug);
        currentDataset = { status: statusInfo.status, isAPI: statusInfo.isAPI };
      }
    } catch (statusError) {
      console.warn(
        "Could not get current dataset status, proceeding with archive request",
      );
    }

    const res = await axios.put(
      `/api/v1/dataset-meta/${idOrSlug}`,
      {
        status: "archived",
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    console.log("✅ Dataset archived successfully:", res.data);
    return {
      success: true,
      data: res.data.data,
      message: "Dataset archived successfully",
    };
  } catch (error: any) {
    console.error("❌ Archive dataset error:", error);

    // Handle status transition validation errors from your route
    if (error.response?.data?.message?.includes("Invalid status transition")) {
      return {
        success: false,
        message: `Cannot archive dataset: ${error.response.data.message}`,
        currentStatus: error.response.data.currentStatus,
        attemptedStatus: error.response.data.attemptedStatus,
        validTransitions: error.response.data.validTransitions,
      };
    }

    throw error;
  }
};

// Reactivate archive dataset function
export const reactivateArchivedDataset = async (collectionName: string) => {
  try {
    const res = await axios.patch(
      `/api/v1/dataset-meta/${collectionName}/reactivate`,
    );

    return {
      success: true,
      data: res.data.data,
      message: res.data.message,
      statusTransition: res.data.statusTransition,
    };
  } catch (error: any) {
    console.error("❌ Reactivate dataset error:", error);

    if (error.response?.data?.message) {
      return {
        success: false,
        message: error.response.data.message,
      };
    }

    throw error;
  }
};

export const canDatasetBeArchived = (
  currentStatus: string,
): { canArchive: boolean; reason?: string } => {
  if (currentStatus === "archived") {
    return {
      canArchive: false,
      reason: "Dataset is already archived",
    };
  }

  return { canArchive: true };
};

export const canDatasetBeRefreshed = (
  currentStatus: string,
  isAPI: boolean = false,
): { canRefresh: boolean; reason?: string } => {
  if (currentStatus === "archived") {
    return {
      canRefresh: false,
      reason: "Archived datasets cannot be refreshed",
    };
  }

  if (isAPI) {
    // API datasets can be refreshed from any non-archived status
    const refreshableStatuses = ["raw", "latest", "preprocessed", "validated"];

    if (!refreshableStatuses.includes(currentStatus)) {
      return {
        canRefresh: false,
        reason: `API dataset with status '${currentStatus}' cannot be refreshed`,
      };
    }
  } else {
    // Non-API datasets have manual refresh (to be implemented)
    const refreshableStatuses = ["raw", "latest", "preprocessed", "validated"];

    if (!refreshableStatuses.includes(currentStatus)) {
      return {
        canRefresh: false,
        reason: `Non-API dataset with status '${currentStatus}' cannot be refreshed`,
      };
    }
  }

  return { canRefresh: true };
};

export const getDatasetRefreshInfo = async (idOrCollectionName: string) => {
  try {
    // First try to get dataset info to determine if it's API or not
    let isAPI = false;
    let collectionName = idOrCollectionName;

    // If it looks like an ObjectId, find the collection name first
    if (idOrCollectionName.match(/^[0-9a-fA-F]{24}$/)) {
      const allDatasets = await GetAllDatasetMeta();
      const dataset = allDatasets.find((d) => d._id === idOrCollectionName);
      if (dataset) {
        isAPI = dataset.isAPI;
        collectionName = dataset.collectionName;
      }
    } else {
      // Get status info to determine API type
      const statusInfo = await getDatasetStatusInfo(idOrCollectionName);
      isAPI = statusInfo.isAPI;
    }

    // Get appropriate refresh status
    const refreshStatus = await getDatasetRefreshStatus(
      isAPI ? idOrCollectionName : collectionName,
      isAPI,
    );

    return {
      isAPI,
      collectionName,
      refreshMethod: isAPI ? "nasa-power-api" : "manual-upload",
      ...refreshStatus,
    };
  } catch (error) {
    console.error("❌ Get dataset refresh info error:", error);
    return {
      isAPI: false,
      collectionName: idOrCollectionName,
      refreshMethod: "unknown",
      canRefresh: false,
      message: "Could not determine dataset refresh information",
    };
  }
};

// Check if dataset can be preprocessed
export const canDatasetBePreprocessed = (
  currentStatus: string,
): { canPreprocess: boolean; reason?: string } => {
  const allowedStatuses = ["raw", "latest"];

  if (!allowedStatuses.includes(currentStatus)) {
    return {
      canPreprocess: false,
      reason: `Cannot preprocess dataset with status '${currentStatus}'. Only 'raw' and 'latest' datasets can be preprocessed.`,
    };
  }

  return { canPreprocess: true };
};

// Function to get dataset status info
export const getDatasetStatusInfo = async (collectionName: string) => {
  try {
    const res = await axios.get(
      `/api/v1/dataset-meta/${collectionName}/status`,
    );
    return res.data;
  } catch (error) {
    console.error("Get dataset status error:", error);
    throw error;
  }
};
export const getDatasetRefreshStatus = async (
  idOrCollectionName: string,
  isAPI: boolean = false,
) => {
  try {
    if (isAPI) {
      // For NASA POWER API datasets, get refresh info
      const baseUrl = getBaseUrl();
      const response = await axios.get(
        `${baseUrl}/api/v1/nasa-power/refreshable`,
      );
      const datasets = response.data.data || [];

      const dataset = datasets.find(
        (d: any) =>
          d._id === idOrCollectionName ||
          d.collectionName === idOrCollectionName,
      );

      if (!dataset) {
        return {
          canRefresh: false,
          message: "Dataset not found",
          status: "unknown",
        };
      }

      return {
        canRefresh: dataset.refreshInfo.canRefresh,
        status: dataset.refreshInfo.statusInfo.current,
        allowsRefresh: dataset.refreshInfo.statusInfo.allowsRefresh,
        willBecomeAfterRefresh:
          dataset.refreshInfo.statusInfo.willBecomeAfterRefresh,
        willDeleteCleanedCollection:
          dataset.refreshInfo.statusInfo.willDeleteCleanedCollection,
        daysSinceLastRecord: dataset.refreshInfo.daysSinceLastRecord,
        lastRecordDate: dataset.refreshInfo.lastRecordDate,
        refreshEligibility: dataset.refreshInfo.refreshEligibility,
        message: dataset.refreshInfo.refreshEligibility,
      };
    } else {
      // For non-API datasets, use dataset-meta status endpoint
      const baseUrl = getBaseUrl();
      const response = await axios.get(
        `${baseUrl}/api/v1/dataset-meta/${idOrCollectionName}/status`,
      );
      return response.data;
    }
  } catch (error) {
    console.error("Error checking dataset refresh status:", error);
    return {
      canRefresh: false,
      message: "Unable to check refresh status",
      status: "unknown",
    };
  }
};

export const UpdateDatasetMeta = async (
  _id: string,
  data: {
    name?: string;
    source?: string;
    fileType?: "csv" | "json";
    filename?: string;
    collectionName?: string;
    status?: string;
    description?: string;
  },
) => {
  try {
    const baseUrl = getBaseUrl();
    const res = await axios.patch(
      `${baseUrl}/api/v1/dataset-meta/${_id}`,
      data,
      {
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    console.log("API response:", res.data);
    return res.data.data;
  } catch (error) {
    console.error("❌ Update dataset meta error:", error);
    throw error;
  }
};

export const fetchNasaPowerData = async (params: NasaPowerParams) => {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(`${baseUrl}/api/v1/nasa-power`, {
      params: {
        start: params.start,
        end: params.end,
        latitude: params.latitude,
        longitude: params.longitude,
        parameters: params.parameters.join(","),
        community: params.community || "ag",
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching NASA Power data:", error);
    throw error;
  }
};
export const saveNasaPowerData = async (data: {
  name: string;
  collectionName?: string;
  description?: string;
  status?: string;
  source?: string;
  nasaParams: NasaPowerParams;
}) => {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.post(
      `${baseUrl}/api/v1/nasa-power/save`,
      data,
    );
    return response.data;
  } catch (error) {
    console.error("Error saving NASA Power data:", error);
    throw error;
  }
};

// Modify refreshNasaPowerDataset to check first
export const refreshNasaPowerDataset = async (datasetId: string) => {
  try {
    // First check if refresh is available and get status info
    const refreshStatus = await getDatasetRefreshStatus(datasetId, true);

    // If dataset status doesn't allow refresh
    if (!refreshStatus.allowsRefresh) {
      return {
        success: false,
        message: `Cannot refresh dataset with status '${refreshStatus.status}'`,
        refreshResult: "status-blocked",
        statusInfo: refreshStatus,
      };
    }

    // If dataset is already up-to-date
    if (!refreshStatus.canRefresh && refreshStatus.allowsRefresh) {
      return {
        success: true,
        message: "Dataset is already up-to-date",
        refreshResult: "up-to-date",
        data: {
          newRecordsCount: 0,
          lastUpdated: refreshStatus.lastRecordDate,
          statusInfo: refreshStatus,
        },
      };
    }

    // Proceed with refresh
    const baseUrl = getBaseUrl();
    const response = await axios.post(
      `${baseUrl}/api/v1/nasa-power/refresh/${datasetId}`,
    );

    return {
      success: true,
      ...response.data,
    };
  } catch (error: any) {
    console.error("Error refreshing NASA POWER dataset:", error);

    // Enhanced error categorization based on your route implementation
    if (error.response?.data?.refreshResult) {
      return {
        success: false,
        message: error.response.data.message,
        refreshResult: error.response.data.refreshResult,
        details: error.response.data.details,
      };
    }

    throw error;
  }
};

// Fetch All NASA datasets
export const refreshAllNasaDatasets = async (): Promise<RefreshAllResponse> => {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.post(
      `${baseUrl}/api/v1/nasa-power/refresh-all`,
    );
    console.log("Full response:", response);
    console.log("response.data:", response.data);
    console.log("response.status:", response.status);
    return response.data;
  } catch (error) {
    console.error("Error refreshing all NASA datasets:", error);
    throw error;
  }
};

export const preprocessNasaDatasetWithStream = (
  collectionName: string,
  onLog: (log: any) => void,
  onProgress: (progress: number, stage: string, message: string) => void,
  onComplete: (result: any) => void,
  onError: (error: string) => void,
) => {
  const baseUrl = getBaseUrl();
  const apiUrl = (
    process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:5001"
  ).replace(/\/$/, "");
  const encodedName = encodeURIComponent(collectionName);

  const abortController = new AbortController();
  let connectionTimeout: NodeJS.Timeout;
  let hasCompletedSuccessfully = false;
  let hasReceivedData = false;
  let streamClosed = false;

  const resetTimeout = () => {
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
    }
    connectionTimeout = setTimeout(() => {
      if (!hasCompletedSuccessfully) {
        onError(
          "Connection timeout - preprocessing may still be running in background",
        );
        abortController.abort();
      }
    }, 300000); // 5 minutes timeout
  };

  const startStream = async () => {
    try {
      resetTimeout();
      const response = await fetch(
        `${apiUrl}/api/v1/preprocess/nasa/${encodedName}/stream`,
        {
          headers: {
            Accept: "text/event-stream",
            "ngrok-skip-browser-warning": "69420",
          },
          signal: abortController.signal,
        },
      );

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No readable stream available");

      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (!streamClosed) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split("\n\n");
        buffer = messages.pop() || "";

        for (const message of messages) {
          if (message.startsWith("data: ")) {
            let dataStr = message.slice(6).trim();
            if (
              !dataStr ||
              dataStr === "keepalive" ||
              dataStr === "completion-sent"
            )
              continue;

            try {
              const data = JSON.parse(dataStr);
              hasReceivedData = true;
              resetTimeout();

              switch (data.type) {
                case "connected":
                  onLog({
                    type: "info",
                    message: `Connected to preprocessing stream. Session: ${data.session_id}`,
                  });
                  break;

                case "log":
                  onLog(data);
                  break;

                case "progress":
                  const progressValue =
                    data.percentage ?? data.progress ?? data.percent ?? 0;
                  onProgress(
                    progressValue,
                    data.stage || "Processing",
                    data.message || "Processing...",
                  );
                  break;

                case "complete":
                  hasCompletedSuccessfully = true;
                  streamClosed = true;

                  onLog({
                    type: "success",
                    message: "🎉 NASA preprocessing completed successfully!",
                  });

                  const completionResult =
                    data.result && typeof data.result === "object"
                      ? data.result
                      : {
                          recordCount: null,
                          cleanedCollection: `${collectionName}_cleaned`,
                          preprocessing_report: null,
                        };

                  onComplete(completionResult);
                  clearTimeout(connectionTimeout);
                  abortController.abort();
                  break;

                case "error":
                  streamClosed = true;
                  onError(
                    data.message || "An error occurred during preprocessing",
                  );
                  clearTimeout(connectionTimeout);
                  abortController.abort();
                  break;

                default:
                  break;
              }
            } catch (err) {
              console.error("Error parsing SSE JSON:", err);
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name === "AbortError" || streamClosed) return;
      clearTimeout(connectionTimeout);
      streamClosed = true;

      if (hasCompletedSuccessfully) return;

      if (hasReceivedData) {
        onLog({
          type: "success",
          message: "✅ Preprocessing completed successfully!",
        });
        onComplete({
          recordCount: null,
          cleanedCollection: `${collectionName}_cleaned`,
          preprocessing_report: null,
        });
        return;
      }

      onError("❌ Connection failed - no preprocessing data received");
    }
  };

  startStream();

  return {
    eventSource: null, // Stub for backward compatibility
    cleanup: () => {
      streamClosed = true;
      clearTimeout(connectionTimeout);
      abortController.abort();
    },
  };
};

// Trigger BMKG Preprocessing with stream
export const preprocessBmkgDatasetWithStream = (
  collectionName: string,
  onLog: (log: any) => void,
  onProgress: (progress: number, stage: string, message: string) => void,
  onComplete: (result: any) => void,
  onError: (error: string) => void,
) => {
  const baseUrl = getBaseUrl();
  const apiUrl = (
    process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:5001"
  ).replace(/\/$/, "");
  const encodedName = encodeURIComponent(collectionName);

  const abortController = new AbortController();
  let connectionTimeout: NodeJS.Timeout;
  let hasCompletedSuccessfully = false;
  let streamClosed = false;

  const resetTimeout = () => {
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
    }
    connectionTimeout = setTimeout(() => {
      onError(
        "Connection timeout - preprocessing may still be running in background",
      );
      abortController.abort();
    }, 300000); // 5 minutes timeout
  };

  const startStream = async () => {
    try {
      resetTimeout();
      const response = await fetch(
        `${apiUrl}/api/v1/preprocess/bmkg/${encodedName}/stream`,
        {
          headers: {
            Accept: "text/event-stream",
            "ngrok-skip-browser-warning": "69420",
          },
          signal: abortController.signal,
        },
      );

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No readable stream available");

      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (!streamClosed) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split("\n\n");
        buffer = messages.pop() || "";

        for (const message of messages) {
          if (message.startsWith("data: ")) {
            let dataStr = message.slice(6).trim();
            if (
              !dataStr ||
              dataStr === "keepalive" ||
              dataStr === "completion-sent"
            )
              continue;

            try {
              const data = JSON.parse(dataStr);
              resetTimeout();

              switch (data.type) {
                case "connected":
                  onLog({
                    type: "info",
                    message: `Connected to BMKG preprocessing stream. Session: ${data.session_id}`,
                  });
                  break;

                case "log":
                  onLog(data);
                  break;

                case "progress":
                  const progressValue =
                    data.percentage ?? data.progress ?? data.percent ?? 0;
                  onProgress(
                    progressValue,
                    data.stage || "Processing",
                    data.message || "Processing...",
                  );
                  break;

                case "complete":
                  hasCompletedSuccessfully = true;
                  streamClosed = true;

                  onLog({
                    type: "success",
                    message: "🎉 BMKG preprocessing completed successfully!",
                  });

                  const completionResult =
                    data.result && typeof data.result === "object"
                      ? data.result
                      : {
                          recordCount: null,
                          cleanedCollection: `${collectionName}_cleaned`,
                          preprocessing_report: null,
                        };

                  onComplete(completionResult);
                  clearTimeout(connectionTimeout);
                  abortController.abort();
                  break;

                case "error":
                  onError(
                    data.message ||
                      "An error occurred during BMKG preprocessing",
                  );
                  abortController.abort();
                  break;

                default:
                  console.log("Unknown SSE message type:", data.type, data);
                  break;
              }
            } catch (err) {
              console.error("Error parsing SSE JSON:", err);
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name === "AbortError" || streamClosed) return;
      console.error("SSE error:", error);
      clearTimeout(connectionTimeout);

      if (!hasCompletedSuccessfully) {
        onError(
          "Connection error occurred with the BMKG preprocessing stream.",
        );
      }
      abortController.abort();
    }
  };

  startStream();

  return {
    eventSource: null, // Stub for backward compatibility
    cleanup: () => {
      streamClosed = true;
      clearTimeout(connectionTimeout);
      abortController.abort();
    },
  };
};

// Trigger BMKG Preprocessing without stream (standard)
export const preprocessBmkgDataset = async (
  collectionName: string,
  options?: {
    smoothing_method?: "exponential" | "moving_average";
    window_size?: number;
    exponential_alpha?: number;
    drop_outliers?: boolean;
    outlier_methods?: string[];
    iqr_multiplier?: number;
    zscore_threshold?: number;
    outlier_treatment?: "interpolate" | "cap" | "remove";
    fill_missing?: boolean;
    detect_gaps?: boolean;
    max_gap_interpolate?: number;
    columns_to_process?: string[];
    parameter_configs?: Record<string, any>;
  },
) => {
  try {
    const response = await axios.post(
      `/api/v1/preprocess/bmkg/${collectionName}`,
      options || {},
    );
    return response.data;
  } catch (error) {
    console.error("Error preprocessing BMKG dataset:", error);
    throw error;
  }
};

// Get BMKG preprocessing job status
export const getBmkgPreprocessingStatus = async (jobId: string) => {
  try {
    const response = await axios.get(`/api/v1/preprocess/status/${jobId}`);
    return response.data;
  } catch (error) {
    console.error("Error getting BMKG preprocessing status:", error);
    throw error;
  }
};

// Get all BMKG preprocessing jobs
export const getAllBmkgPreprocessingJobs = async (page = 1, pageSize = 10) => {
  try {
    const response = await axios.get("/api/v1/preprocess/jobs", {
      params: {
        page,
        pageSize,
        type: "bmkg", // Filter for BMKG jobs only
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error getting BMKG preprocessing jobs:", error);
    throw error;
  }
};

export const updateDatasetStatus = async (
  idOrSlug: string,
  newStatus: string,
  additionalData?: Record<string, any>,
) => {
  try {
    const updateData = {
      status: newStatus,
      ...additionalData,
    };

    const res = await axios.put(
      `/api/v1/dataset-meta/${idOrSlug}`,
      updateData,
      {
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    console.log("✅ Dataset status updated successfully:", res.data);
    return {
      success: true,
      data: res.data.data,
      message: res.data.message,
    };
  } catch (error: any) {
    console.error("❌ Update dataset status error:", error);

    // Handle status transition validation errors
    if (error.response?.data?.message?.includes("Invalid status transition")) {
      return {
        success: false,
        message: error.response.data.message,
        currentStatus: error.response.data.currentStatus,
        attemptedStatus: error.response.data.attemptedStatus,
        validTransitions: error.response.data.validTransitions,
        transitionError: true,
      };
    }

    throw error;
  }
};

export const checkDatasetOperations = async (collectionName: string) => {
  try {
    const statusInfo = await getDatasetStatusInfo(collectionName);

    return {
      collectionName: statusInfo.collectionName,
      currentStatus: statusInfo.status,
      isAPI: statusInfo.isAPI,
      operations: {
        canPreprocess: statusInfo.canPreprocess,
        canRefresh: statusInfo.canRefresh,
        canArchive: statusInfo.status !== "archived",
        canReactivate: statusInfo.canReactivate,
      },
      hasCleanedCollection: statusInfo.hasCleanedCollection,
      operationDetails: statusInfo.operations,
    };
  } catch (error) {
    console.error("❌ Check dataset operations error:", error);
    return {
      collectionName,
      currentStatus: "unknown",
      isAPI: false,
      operations: {
        canPreprocess: false,
        canRefresh: false,
        canArchive: false,
        canReactivate: false,
      },
      hasCleanedCollection: false,
      operationDetails: {},
      error: "Could not determine operation eligibility",
    };
  }
};

export const validateStatusTransition = (
  currentStatus: string,
  newStatus: string,
): {
  isValid: boolean;
  reason?: string;
  validTransitions?: string[];
} => {
  const validTransitions: Record<string, string[]> = {
    raw: ["latest", "preprocessed", "archived"],
    latest: ["raw", "preprocessed", "archived"],
    preprocessed: ["raw", "validated", "archived"],
    validated: ["raw", "archived"],
    archived: ["raw", "latest"], // Reactivation
  };

  const allowedTransitions = validTransitions[currentStatus];

  if (!allowedTransitions) {
    return {
      isValid: false,
      reason: `Unknown current status: ${currentStatus}`,
      validTransitions: [],
    };
  }

  if (!allowedTransitions.includes(newStatus)) {
    return {
      isValid: false,
      reason: `Invalid transition from ${currentStatus} to ${newStatus}`,
      validTransitions: allowedTransitions,
    };
  }

  return { isValid: true };
};
export const refreshNonApiDataset = async (collectionName: string) => {
  try {
    const response = await axios.post(
      `/api/v1/dataset-meta/${collectionName}/refresh`,
    );

    return {
      success: true,
      ...response.data,
    };
  } catch (error: any) {
    console.error("❌ Non-API refresh error:", error);

    // Handle the 501 Not Implemented response from your route
    if (error.response?.status === 501) {
      return {
        success: false,
        notImplemented: true,
        message: error.response.data.message,
        note: error.response.data.note,
        implementation: error.response.data.implementation,
      };
    }

    throw error;
  }
};
export const getNasaPowerRefreshableDatasets = async () => {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(
      `${baseUrl}/api/v1/nasa-power/refreshable`,
    );
    return response.data;
  } catch (error) {
    console.error("Error getting NASA Power refreshable datasets:", error);
    throw error;
  }
};

export const AddXlsxDatasetMeta = async (data: {
  name: string;
  source: string;
  description?: string;
  status?: string;
  file?: File;
  files?: File[];
  isMultiFile?: boolean;
}) => {
  try {
    // Validation: Must provide either file or files
    if (data.isMultiFile) {
      if (!data.files || data.files.length === 0) {
        throw new Error("Files array is required for multi-file upload");
      }
      if (data.files.length > 50) {
        throw new Error("Maximum 50 files allowed for multi-file upload");
      }
    } else {
      if (!data.file) {
        throw new Error("File is required for single file upload");
      }
    }

    // Create FormData for multipart upload
    const formData = new FormData();
    formData.append("name", data.name.trim());
    formData.append("source", data.source.trim());
    formData.append("description", data.description || "");
    formData.append("status", data.status || "raw");
    formData.append("isMultiFile", data.isMultiFile ? "true" : "false");

    // Add files based on upload mode
    if (data.isMultiFile && data.files) {
      // Multi-file upload: add all files with "files" key
      data.files.forEach((file) => {
        formData.append("files", file);
      });
    } else if (data.file) {
      // Single-file upload: add single file with "file" key
      formData.append("file", data.file);
    }

    const baseUrl = getBaseUrl();
    const res = await axios.post(`${baseUrl}/api/v1/dataset-meta`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res.data.data;
  } catch (error) {
    console.error("Add XLSX dataset meta error:", error);
    throw error;
  }
};

// Get preprocessing report by ID
export const getPreprocessingReportById = async (
  id: string,
): Promise<PreprocessingReport> => {
  try {
    const baseUrl = getBaseUrl();
    const res = await axios.get(`${baseUrl}/api/v1/preprocessing-report/${id}`);
    return res.data.data;
  } catch (error) {
    console.error("Error fetching preprocessing report:", error);
    throw error;
  }
};

// Decomposition fetching
export const getDecompositionByPreprocessingId = async (
  id: string,
): Promise<DecompositionReport> => {
  try {
    const baseUrl = getBaseUrl();
    const res = await axios.get(
      `${baseUrl}/api/v1/preprocessing-report/${id}/decomposition`, // Fixed to match your backend route
    );
    return res.data.data;
  } catch (error) {
    console.error("Error fetching decomposition report:", error);
    throw error;
  }
};
