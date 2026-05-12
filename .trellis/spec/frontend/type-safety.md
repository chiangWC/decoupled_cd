# Type Safety

> Frontend TypeScript conventions are not applicable yet.

---

## Current State

The project uses Python type hints and PyTorch tensor contracts, not TypeScript.

For current work, follow backend conventions:

- Use type hints on function signatures.
- Use dataclasses for shared structured data.
- Validate runtime data contracts at boundaries, especially CSV schemas, graph modes, training modes, and history visibility.

---

## Future Rule

If TypeScript is introduced later, document real type organization after the first frontend implementation exists.
