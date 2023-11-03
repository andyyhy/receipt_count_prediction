document.addEventListener("DOMContentLoaded", function () {});

document.getElementById("submit-button").addEventListener("click", function () {
	var form = document.querySelector("form");
	// Perform client-side validation
	var startDate = document.getElementById("start-date").value;
	var endDate = document.getElementById("end-date").value;
	if (new Date(startDate) > new Date(endDate)) {
		alert("Start date must be before end date.");
	} else if (new Date(startDate) < new Date("2021-12-31")) {
		alert("Please pick a start date after 2022-1-1.");
	} else if (startDate === "" || endDate === "") {
		alert("Please select a start and end date.");
	} else {
		fetch("/predict", {
			// Your Flask route
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ start_date: startDate, end_date: endDate }),
		})
			.then((response) => response.json())
			.then((data) => {
				// Assuming the server responds with the data to plot
				if (data.success) {
					const layout = {
						title: "Daily Scanned Receipts over Time", // The plot title
						xaxis: {
							title: "Date", // Label for the x-axis
						},
						yaxis: {
							title: "Number of Receipts", // Label for the y-axis
						},
						legend: {
							title: {
								text: "Legend", // Title for the legend (optional, can be omitted)
							},
						},
						showlegend: true, // Specify whether to show the legend (default is true)
					};
					const config = {
						responsive: true,
					};
					plotData(
						data.predictedData,
						data.originalData,
						layout,
						config
					); // Call a function to handle Plotly plotting

					displayMonthlyReceipts(data.monthlyReceipts);
				} else {
					console.log(data.error);
				}
			})
			.catch((error) => {
				console.error("Error:", error);
				statusDiv.textContent =
					"An error occurred while uploading the file.";
			});
	}
});

function plotData(plotData1, plotData2, layout, config) {
	//

	// Using Plotly to plot the data
	Plotly.newPlot("plot", [plotData1, plotData2], layout, config);
}

function displayMonthlyReceipts(totals) {
	const list = document.getElementById("monthly-totals-list");
	list.innerHTML = ""; // Clear existing list items if any
	console.log(totals);
	// Create list items for each predicted total and add to the list
	totals.forEach((total) => {
		console.log(total.key);
		const listItem = document.createElement("li");
		listItem.textContent = `${total[0]}: ${total[1]} Receipts`;
		list.appendChild(listItem);
	});
}
