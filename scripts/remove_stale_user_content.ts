/**
 * Remove stale user content from R2.
 *
 * Requires npm package @aws-sdk/client-s3.
 */

const R2_ENDPOINT = "<FILL IN R2 ENDPOINT HERE>";
const BUCKET_NAME = "mlir-playground-ugc";
const STALE_TIME_THRESHOLD = 48 * 60 * 60 * 1000 // 48 hours in milliseconds

import {
  S3Client,
  ListObjectsV2Command,
  ListObjectsV2CommandInput,
  DeleteObjectsCommandInput,
  ObjectIdentifier,
  DeleteObjectsCommand,
} from "@aws-sdk/client-s3";

const client = new S3Client({
  region: "auto",
  endpoint: R2_ENDPOINT,
});

async function removeStaleUserContents(staleBeforeTime: number) {
  let staleObjectKeys: Array<string> = [];

  // Find all stale objects' keys.
  let truncated = true;
  let listObjectsParams: ListObjectsV2CommandInput = {
    Bucket: BUCKET_NAME
  };
  while (truncated) {
    try {
      const listResponse = await client.send(new ListObjectsV2Command(listObjectsParams));

      listResponse.Contents?.forEach((item) => {
        if (item.LastModified && item.Key && item.LastModified.getTime() < staleBeforeTime) {
          staleObjectKeys.push(item.Key);
        }
      })

      truncated = !!listResponse.IsTruncated;
      if (truncated) {
        listObjectsParams.ContinuationToken = listResponse.NextContinuationToken;
      }
    } catch (error) {
      console.log("Failed to list objects: " + String(error));
      return;
    }
  }

  console.log("Found %d stale objects.", staleObjectKeys.length);

  // Delete stale keys from the bucket.
  while (staleObjectKeys.length > 0) {
    // Get the first 1000 keys, and remove them from the original list.
    const currentBatchObjects: Array<ObjectIdentifier> = staleObjectKeys.splice(0, 1000).map((key) => ({Key: key}));
    const deleteObjectsParams: DeleteObjectsCommandInput = {
      Bucket: BUCKET_NAME,
      Delete: {
        Objects: currentBatchObjects,
        Quiet: true
      }
    };

    try {
      const deleteResponse = await client.send(new DeleteObjectsCommand(deleteObjectsParams));

      if (deleteResponse.Errors && deleteResponse.Errors.length > 0) {
        console.log("Delete request returned errors:");
        deleteResponse.Errors.forEach((error) => {
          console.log(error.Message!);
        });
        return;
      }
    } catch (error) {
      console.log("Failed to delete objects: " + String(error));
      return;
    }
  }
}

removeStaleUserContents(Date.now() - STALE_TIME_THRESHOLD);
