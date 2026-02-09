# TRK Tire Sorter

A Streamlit web application for sorting and managing tire data for Trackhouse Racing.

## Features

- Import tire data from Excel files
- Drag-and-drop sorting interface
- Multi-set management
- Export sorted data back to Excel
- Custom tire grouping and organization

## Live App

🔗 **[Launch TRK Tire Sorter](https://tiresort.streamlit.app)** *(Update this URL after deployment)*

## Local Installation

If you prefer to run the app locally:

```bash
# Clone the repository
git clone https://github.com/chavely-99/tiresort.git
cd tiresort

# Install dependencies
pip install -r requirements_cloud.txt

# Run the app
streamlit run TireSorter_Streamlit_v1.py
```

## For Trackhouse Team Members

The app is hosted on Streamlit Cloud and accessible via your browser - no installation required!

### How to Use

1. Navigate to the live app URL above
2. Upload your Excel file with tire data
3. Use the drag-and-drop interface to organize tires
4. Export your sorted data as Excel

## Technical Stack

- **Framework**: Streamlit
- **UI Components**: streamlit-sortables for drag-and-drop
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Altair
- **File Handling**: openpyxl for Excel support

## Support

For issues or questions, contact the Trackhouse Racing data team.

---

Built with ❤️ for Trackhouse Racing
