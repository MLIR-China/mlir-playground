/**
 * Script for deleting preview deployments of this PR on Cloudflare pages.
 * Used by the github actions when a PR is closed.
 */

export {};

if (process.argv.length != 6) {
  console.error("Expected 4 command line arguments.");
  console.log(
    "Usage: npx ts-node delete_preview_deployments.ts <CLOUDFLARE_API_TOKEN> <CLOUDFLARE_ACCOUNT_ID> <CLOUDFLARE_PROJECT_NAME> <BRANCH_NAME>"
  );
  process.exit(1);
}

// First 2 args are "ts-node" and the script file respectively.
const CLOUDFLARE_API_TOKEN = process.argv[2];
const CLOUDFLARE_ACCOUNT_ID = process.argv[3];
const CLOUDFLARE_PROJECT_NAME = process.argv[4];
const BRANCH_NAME = process.argv[5];

const REQUEST_HEADERS = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${CLOUDFLARE_API_TOKEN}`,
};

// Fetch all deployments matching the branch name
async function getPreviewDeploymentsOfBranch(branch: string) {
  let matchingDeploymentIds: Array<string> = [];

  const fetchUrl = `https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/pages/projects/${CLOUDFLARE_PROJECT_NAME}/deployments`;
  let processedCount = 0;
  let pageIndex = 1;
  while (true) {
    const deploymentsResponse: any = await (
      await fetch(fetchUrl + `?page=${pageIndex}`, {
        method: "GET",
        headers: REQUEST_HEADERS,
      })
    ).json();

    if (!deploymentsResponse.success) {
      console.log("Error fetching deployments.");
      console.log(deploymentsResponse.errors);
      process.exit(1);
    }

    deploymentsResponse.result.forEach((deployment: any) => {
      if (
        deployment.environment === "preview" &&
        deployment.deployment_trigger.metadata.branch === branch
      ) {
        matchingDeploymentIds.push(deployment.id);
      }
    });

    processedCount += deploymentsResponse.result.length;

    if (deploymentsResponse.result_info.total_count > processedCount) {
      pageIndex++;
    } else {
      break;
    }
  }

  console.log(
    "Found %d matching preview deployments.",
    matchingDeploymentIds.length
  );

  // Reverse list so that the oldest deployment comes first.
  return matchingDeploymentIds.reverse();
}

async function deleteDeployments(deploymentIds: Array<string>) {
  const deleteUrl = `https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/pages/projects/${CLOUDFLARE_PROJECT_NAME}/deployments/`;
  let successCount = 0;
  for (const deploymentId of deploymentIds) {
    const deleteResponse = await (
      await fetch(deleteUrl + deploymentId, {
        method: "DELETE",
        headers: REQUEST_HEADERS,
      })
    ).json();

    if (!deleteResponse.success) {
      console.log("Error deleting deployment: " + deploymentId);
      console.log(deleteResponse.errors);
    } else {
      console.log("Deleted: " + deploymentId);
      successCount++;
    }
  }

  console.log("Successfully deleted %d deployments.", successCount);
}

getPreviewDeploymentsOfBranch(BRANCH_NAME).then((previewDeploymentIds) => {
  deleteDeployments(previewDeploymentIds);
});
