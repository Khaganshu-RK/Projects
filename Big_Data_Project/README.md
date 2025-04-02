# **Azure Big Data Pipeline - README**

## Project Overview

This project implements a **big data pipeline** using **Azure Data Factory, ADLS Gen2, Databricks, Synapse, MongoDB, PostgreSQL, and HTTPS API**. The pipeline follows the **Medallion Architecture** (Bronze → Silver → Gold) for efficient data processing and analytics.

---

## Project Architecture

1. **Data Ingestion (Bronze Layer - Raw Data Storage)**

   - Data sources:
     - **GitHub API / HTTPS Endpoint** (for raw JSON data)
     - **SQL Data (PostgreSQL)** (structured data)
   - Tools Used: **Azure Data Factory (ADF), ADLS Gen2**
   - Process:
     - Used **`ForEach` activity** in ADF to fetch multiple files from GitHub API.
     - Used **Copy Activity** in ADF to extract SQL data and store it as CSV.
     - Stored raw data in **ADLS Gen2 - Bronze folder**.

2. **Data Processing (Silver Layer - Enriched Data)**

   - Tools Used: **Azure Databricks (PySpark), MongoDB**
   - Process:
     - Read data from **ADLS Gen2 - Bronze folder**.
     - Cleaned and transformed data using **PySpark in Databricks**.
     - Fetched **additional enriched data from MongoDB** in Databricks.
     - Combined and transformed data.
     - Stored the cleaned and structured data in **ADLS Gen2 - Silver folder (Parquet format)**.

3. **Data Aggregation & Analytics (Gold Layer - Analytics Ready Data)**
   - Tools Used: **Azure Synapse Analytics**
   - Process:
     - Synapse ingested the **Silver-layer data from ADLS Gen2**.
     - Performed **further transformations & aggregations**.
     - Stored the final analytics-ready dataset in **ADLS Gen2 - Gold folder** as **CETAS Parquet tables**.
     - Data Science & Data Analysts can now query the Gold-layer data efficiently.

---

## Technologies Used

| Tool/Technology                         | Purpose                                  |
| --------------------------------------- | ---------------------------------------- |
| **Azure Data Factory (ADF)**            | Data ingestion from APIs and SQL         |
| **Azure Data Lake Storage (ADLS Gen2)** | Storage following Medallion architecture |
| **Azure Databricks (PySpark)**          | Data transformation and enrichment       |
| **MongoDB**                             | Additional data source for enrichment    |
| **PostgreSQL**                          | Structured data source                   |
| **Azure Synapse Analytics**             | Data transformation & analytics          |
| **Parquet Format (CETAS)**              | Optimized data storage                   |

---

## Steps to Reproduce

### **Step 1: Data Ingestion**

1. **Create an Azure Data Factory Pipeline**:
   - Use **ForEach** loop to fetch data from **GitHub API**.
   - Use **Copy Activity** to extract data from **PostgreSQL**.
   - Store raw data in **ADLS Gen2 - Bronze Layer**.

### **Step 2: Data Processing & Enrichment**

2. **Use Databricks (PySpark) for transformation**:
   - Read raw data from **Bronze folder**.
   - Clean & transform data using **PySpark**.
   - Fetch **enriched data from MongoDB**.
   - Merge all datasets and store in **ADLS Gen2 - Silver Layer (Parquet Format)**.

### **Step 3: Data Aggregation & Storage**

3. **Use Azure Synapse Analytics for final processing**:
   - Read Silver data from **ADLS Gen2**.
   - Perform additional transformations & aggregations.
   - Store results in **ADLS Gen2 - Gold Layer (CETAS Parquet Format)**.
   - Data Analysts & Scientists use this optimized data for analysis.

---

## Medallion Architecture in This Project

| Layer      | Storage                   | Purpose                                      |
| ---------- | ------------------------- | -------------------------------------------- |
| **Bronze** | ADLS Gen2 (Raw Data)      | Stores raw, unprocessed data from APIs & SQL |
| **Silver** | ADLS Gen2 (Parquet)       | Cleaned, transformed, and enriched data      |
| **Gold**   | ADLS Gen2 (CETAS Parquet) | Aggregated & analytics-ready data            |

---

## Key Takeaways

**Scalable Data Pipeline** using Azure services.  
**Efficient Storage** using ADLS Gen2 with Parquet format.  
**Seamless Data Integration** with multiple sources (API, SQL, MongoDB).  
**Performance Optimization** using Synapse & CETAS for analytical workloads.

---

## Future Enhancements

- Automate Synapse queries using **Azure Logic Apps / Azure Functions**.
- Implement **Data Quality Checks** in Databricks.
- Enable **Real-time Processing** using Event Hub & Streaming Analytics.

---
