import json
from pathlib import Path
from harness.api.main import create_app

app = create_app()
schema = app.openapi()

out_lines = []
out_lines.append("# HarnessAgent API Reference\n")
out_lines.append(f"{schema.get('info', {}).get('description', '')}\n")

paths = schema.get("paths", {})
for path, methods in paths.items():
    for method, details in methods.items():
        summary = details.get('summary', 'Endpoint')
        out_lines.append(f"## {method.upper()} {path}")
        out_lines.append(f"**{summary}**\n")
        if 'description' in details:
            out_lines.append(f"{details['description']}\n")
        
        # Parameters
        if 'parameters' in details:
            out_lines.append("### Parameters\n")
            out_lines.append("| Name | In | Required | Type | Description |")
            out_lines.append("|---|---|---|---|---|")
            for param in details['parameters']:
                name = param.get('name', '')
                in_ = param.get('in', '')
                req = "Yes" if param.get('required') else "No"
                schema_type = param.get('schema', {}).get('type', 'string')
                desc = param.get('description', '').replace("\n", " ")
                out_lines.append(f"| `{name}` | {in_} | {req} | {schema_type} | {desc} |")
            out_lines.append("")
        
        # Request Body
        if 'requestBody' in details:
            out_lines.append("### Request Body\n")
            content = details['requestBody'].get('content', {})
            if 'application/json' in content:
                ref = content['application/json'].get('schema', {}).get('$ref', '')
                if ref:
                    model_name = ref.split("/")[-1]
                    out_lines.append(f"Requires `{model_name}` JSON object.\n")
        
        # Responses
        if 'responses' in details:
            out_lines.append("### Responses\n")
            out_lines.append("| Code | Description |")
            out_lines.append("|---|---|")
            for code, resp in details['responses'].items():
                desc = resp.get('description', '')
                out_lines.append(f"| {code} | {desc} |")
            out_lines.append("")

out_lines.append("## Schemas\n")
schemas = schema.get("components", {}).get("schemas", {})
for model_name, model_details in schemas.items():
    out_lines.append(f"### {model_name}")
    if 'description' in model_details:
        out_lines.append(f"{model_details['description']}\n")
    out_lines.append("| Property | Type | Description |")
    out_lines.append("|---|---|---|")
    for prop, prop_details in model_details.get('properties', {}).items():
        ptype = prop_details.get('type', 'string')
        if '$ref' in prop_details:
            ptype = f"reference to {prop_details['$ref'].split('/')[-1]}"
        elif 'anyOf' in prop_details:
            ptype = "union"
        pdesc = prop_details.get('description', '').replace("\n", " ")
        out_lines.append(f"| `{prop}` | {ptype} | {pdesc} |")
    out_lines.append("")

doc_path = Path("docs/reference/API_REFERENCE.md")
doc_path.write_text("\n".join(out_lines))
print(f"API Reference written to {doc_path}")
