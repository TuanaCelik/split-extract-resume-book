import { useEffect, useState, useMemo } from "react";
import {
  AcceptReject,
  ExtractedDataDisplay,
  FilePreview,
  useItemData,
  type Highlight,
  type ExtractedData,
  Button,
} from "@llamaindex/ui";
import { Clock, XCircle, Download } from "lucide-react";
import { useParams } from "react-router-dom";
import { useToolbar } from "@/lib/ToolbarContext";
import { useNavigate } from "react-router-dom";
import { modifyJsonSchema } from "@llamaindex/ui/lib";
import { APP_TITLE } from "@/lib/config";
import { downloadExtractedDataItem } from "@/lib/export";
import { useMetadataContext } from "@/lib/MetadataProvider";
import { convertBoundingBoxesToHighlights } from "@/lib/utils";

/**
 * Select the appropriate schema based on the discriminator field value.
 * If multiple schemas are provided and the item has a discriminator value,
 * use the type-specific schema for a focused editing experience.
 * Otherwise, fall back to the union schema.
 */
function selectSchemaForItem(
  metadata: {
    json_schema: any;
    schemas?: Record<string, any>;
    discriminator_field?: string;
  },
  itemData: any,
): any {
  const { schemas, discriminator_field, json_schema } = metadata;

  // If no discriminator support, use the union schema
  if (!schemas || !discriminator_field) {
    return json_schema;
  }

  // Get the discriminator value from the extracted data
  // item.data contains wrapper fields (status, file_id, etc.)
  // item.data.data contains the actual extracted fields including the discriminator
  const discriminatorValue = itemData?.data?.data?.[discriminator_field];

  // If we have a valid discriminator value and a matching schema, use it
  if (discriminatorValue && schemas[discriminatorValue]) {
    return schemas[discriminatorValue];
  }

  // Fall back to the union schema
  return json_schema;
}

export default function ItemPage() {
  const { itemId } = useParams<{ itemId: string }>();
  const { setButtons, setBreadcrumbs } = useToolbar();
  const [highlight, setHighlight] = useState<Highlight | undefined>(undefined);
  const { metadata } = useMetadataContext();
  // Use the hook to fetch item data
  const itemHookData = useItemData<any>({
    // order/remove fields as needed here
    jsonSchema: modifyJsonSchema(metadata.json_schema, {}),
    itemId: itemId as string,
    isMock: false,
  });

  // Select the appropriate schema based on discriminator field
  const selectedSchema = useMemo(() => {
    return selectSchemaForItem(metadata, itemHookData.item);
  }, [metadata, itemHookData.item]);

  // Modify the selected schema for display
  const displaySchema = useMemo(() => {
    return modifyJsonSchema(selectedSchema, {});
  }, [selectedSchema]);

  const navigate = useNavigate();

  // Update breadcrumb when item data loads
  useEffect(() => {
    const extractedData = itemHookData.item?.data as
      | ExtractedData<unknown>
      | undefined;
    const fileName = extractedData?.file_name;
    if (fileName) {
      setBreadcrumbs([
        { label: APP_TITLE, href: "/" },
        {
          label: fileName,
          isCurrentPage: true,
        },
      ]);
    }

    return () => {
      // Reset to default breadcrumb when leaving the page
      setBreadcrumbs([{ label: APP_TITLE, href: "/" }]);
    };
  }, [itemHookData.item?.data, setBreadcrumbs]);

  useEffect(() => {
    setButtons(() => [
      <div className="ml-auto flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            if (itemData) {
              downloadExtractedDataItem(itemData);
            }
          }}
          disabled={!itemData}
          startIcon={<Download className="h-4 w-4" />}
          label="Export JSON"
        />
        <AcceptReject<any>
          itemData={itemHookData}
          onComplete={() => navigate("/")}
        />
      </div>,
    ]);
    return () => {
      setButtons(() => []);
    };
  }, [itemHookData.data, setButtons]);

  const {
    item: itemData,
    updateData,
    loading: isLoading,
    error,
  } = itemHookData;

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <Clock className="h-8 w-8 animate-spin mx-auto mb-2" />
          <div className="text-sm text-gray-500">Loading item...</div>
        </div>
      </div>
    );
  }

  if (error || !itemData) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <XCircle className="h-8 w-8 text-red-500 mx-auto mb-2" />
          <div className="text-sm text-gray-500">
            Error loading item: {error || "Item not found"}
          </div>
        </div>
      </div>
    );
  }

  // Cast itemData.data to ExtractedData for proper typing
  const extractedData = itemData.data as ExtractedData<any>;
  const fileId = extractedData.file_id;

  return (
    <div className="flex h-full bg-gray-50">
      {/* Left Side - File Preview */}
      <div className="w-1/2 border-r h-full border-gray-200 bg-white">
        {fileId && (
          <FilePreview
            fileId={fileId}
            onBoundingBoxClick={(box, pageNumber) => {
              console.log("Bounding box clicked:", box, "on page:", pageNumber);
            }}
            highlight={highlight}
          />
        )}
      </div>

      {/* Right Side - Review Panel */}
      <div className="flex-1 bg-white h-full overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Extracted Data */}
          <ExtractedDataDisplay<any>
            extractedData={extractedData}
            title="Extracted Data"
            onChange={(updatedData) => {
              updateData(updatedData);
            }}
            onHoverField={(args) => {
              const highlights = convertBoundingBoxesToHighlights(
                args?.metadata?.citation,
              );
              setHighlight(highlights[0]);
            }}
            jsonSchema={displaySchema}
          />
        </div>
      </div>
    </div>
  );
}
