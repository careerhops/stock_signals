(function () {
  function parseCellValue(cell) {
    const raw = (cell.textContent || "").trim();
    if (!raw) {
      return { type: "empty", value: "" };
    }

    const normalized = raw.replace(/,/g, "").replace(/%/g, "");
    const numberValue = Number(normalized);
    if (!Number.isNaN(numberValue) && normalized !== "") {
      return { type: "number", value: numberValue };
    }

    const dateValue = Date.parse(raw);
    if (!Number.isNaN(dateValue) && /^\d{4}-\d{2}-\d{2}/.test(raw)) {
      return { type: "date", value: dateValue };
    }

    return { type: "text", value: raw.toLowerCase() };
  }

  function compareValues(a, b, direction) {
    if (a.type === "empty" && b.type !== "empty") return 1;
    if (b.type === "empty" && a.type !== "empty") return -1;
    if (a.value < b.value) return direction === "asc" ? -1 : 1;
    if (a.value > b.value) return direction === "asc" ? 1 : -1;
    return 0;
  }

  function initialDirectionForColumn(rows, columnIndex) {
    const firstValue = rows.map((row) => parseCellValue(row.cells[columnIndex])).find((value) => value.type !== "empty");
    return firstValue && (firstValue.type === "number" || firstValue.type === "date") ? "desc" : "asc";
  }

  function sortTable(table, columnIndex, header) {
    const tbody = table.tBodies[0];
    if (!tbody) return;

    const rows = Array.from(tbody.rows);
    const currentDirection = header.getAttribute("data-sort-direction");
    const nextDirection = currentDirection
      ? currentDirection === "asc" ? "desc" : "asc"
      : initialDirectionForColumn(rows, columnIndex);

    rows.sort((rowA, rowB) => {
      const valueA = parseCellValue(rowA.cells[columnIndex]);
      const valueB = parseCellValue(rowB.cells[columnIndex]);
      return compareValues(valueA, valueB, nextDirection);
    });

    table.querySelectorAll("th").forEach((th) => {
      th.removeAttribute("data-sort-direction");
      th.removeAttribute("aria-sort");
    });
    header.setAttribute("data-sort-direction", nextDirection);
    header.setAttribute("aria-sort", nextDirection === "asc" ? "ascending" : "descending");
    rows.forEach((row) => tbody.appendChild(row));
  }

  function enableTableSorting() {
    document.querySelectorAll("table").forEach((table) => {
      const headers = table.querySelectorAll("thead th");
      headers.forEach((header, columnIndex) => {
        header.tabIndex = 0;
        header.classList.add("sortable-header");
        header.addEventListener("click", () => sortTable(table, columnIndex, header));
        header.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            sortTable(table, columnIndex, header);
          }
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", enableTableSorting);
})();
