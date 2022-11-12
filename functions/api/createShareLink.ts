import { validateAgainstSchema } from "../../schema/validation";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,HEAD,POST,OPTIONS",
  "Access-Control-Max-Age": "86400",
};

const USER_CONTENT_SIZE_LIMIT = 128 * 2 ** 10; // 128KB

function createUserErrorResponse(message: string) {
  return new Response(JSON.stringify({ error: message }), {
    status: 400,
  });
}

function buf2hex(buffer: ArrayBuffer): string {
  return [...new Uint8Array(buffer)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function handleRequest(request: Request, env: any) {
  /// Legalize incoming data.
  const readResult: { success: boolean; result: string } = await request
    .text()
    .then(
      (result) => {
        return { success: true, result: result };
      },
      (failure) => {
        return { success: false, result: failure };
      }
    );

  // Check: body can be read as string.
  if (!readResult.success) {
    return createUserErrorResponse(
      "Failed to parse read incoming data: " + readResult.result
    );
  }

  // Check: size is within limit
  const encoder = new TextEncoder();
  const encodedData = encoder.encode(readResult.result);
  if (encodedData.length > USER_CONTENT_SIZE_LIMIT) {
    return createUserErrorResponse("Data size exceeds limit.");
  }

  // Check: body is JSON.
  let parsedResult = null;
  try {
    parsedResult = JSON.parse(readResult.result);
  } catch (error: any) {
    return createUserErrorResponse(
      "Failed to parse incoming data as JSON. " + error.message
    );
  }

  // Check: data conforms to latest schema
  let errorMsg = validateAgainstSchema(parsedResult);
  if (errorMsg) {
    return createUserErrorResponse(
      "Incoming data does not conform to latest schema: " + errorMsg
    );
  }

  /// Everything legal. Save into R2.
  const hashBuffer = await crypto.subtle.digest("SHA-256", encodedData);
  const filename = buf2hex(hashBuffer) + ".json";
  await env.UGC_BUCKET.put(filename, readResult.result);

  return new Response(
    JSON.stringify({
      resource: env.UGC_URL_PREFIX + filename,
    }),
    {
      headers: CORS_HEADERS,
    }
  );
}

function handlePreflightOptions(request) {
  // Make sure the necessary headers are present
  // for this to be a valid pre-flight request
  let headers = request.headers;
  if (
    headers.get("Origin") !== null &&
    headers.get("Access-Control-Request-Method") !== null &&
    headers.get("Access-Control-Request-Headers") !== null
  ) {
    // Handle CORS pre-flight request.
    // If you want to check or reject the requested method + headers
    // you can do that here.
    let respHeaders = {
      ...CORS_HEADERS,
      // Allow all future content Request headers to go back to browser
      // such as Authorization (Bearer) or X-Client-Name-Version
      "Access-Control-Allow-Headers": request.headers.get(
        "Access-Control-Request-Headers"
      ),
    };

    return new Response(null, {
      headers: respHeaders,
    });
  } else {
    // Handle standard OPTIONS request.
    // If you want to allow other HTTP Methods, you can do that here.
    return new Response(null, {
      headers: {
        Allow: "GET, HEAD, POST, OPTIONS",
      },
    });
  }
}

export async function onRequest(context: any) {
  // Contents of context object
  const {
    request, // same as existing Worker API
    env, // same as existing Worker API
    params, // if filename includes [id] or [[path]]
    waitUntil, // same as ctx.waitUntil in existing Worker API
    next, // used for middleware or to fetch assets
    data, // arbitrary space for passing data between middlewares
  } = context;

  if (request.method === "OPTIONS") {
    return handlePreflightOptions(request);
  }

  return handleRequest(request, env);
}
