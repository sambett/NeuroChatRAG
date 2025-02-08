import os
import time
import logging
from Bio import Entrez
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parkinson_resource_log.txt'),
        logging.StreamHandler()
    ]
)

# Configure Entrez properly
Entrez.email = "ramybrahim66@gmail.com"  # MUST REPLACE WITH YOUR EMAIL
Entrez.api_key = "8ef4eaf5889993fd2ea3edda04d31fe1d609"  # Get from NCBI account (optional but recommended)
Entrez.sleep_between_tries = 15  # Increased from default

def create_folders():
    folders = [
        "Parkinson_Resources/PubMed_Articles/PDFs",
        "Parkinson_Resources/PubMed_Articles/Abstracts",
        "Parkinson_Resources/Clinical_Guidelines/NICE",
        "Parkinson_Resources/Clinical_Guidelines/Michael_J_Fox_Foundation",
        "Parkinson_Resources/Public_Datasets/PPMI"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def fetch_pubmed_articles(search_term, max_articles=10):
    """Fetch PubMed articles using esearch and efetch (without using the deprecated egquery)"""
    try:
        # Directly use esearch to obtain a list of article IDs
        handle = Entrez.esearch(
            db="pubmed",
            term=search_term,
            retmax=max_articles,
            usehistory="y"
        )
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            logging.warning("No articles found with current search parameters")
            return

        webenv = record["WebEnv"]
        query_key = record["QueryKey"]

        # Fetch detailed records using efetch
        fetch_handle = Entrez.efetch(
            db="pubmed",
            retmode="xml",
            webenv=webenv,
            query_key=query_key
        )
        articles = Entrez.read(fetch_handle)["PubmedArticle"]
        fetch_handle.close()

        for i, article in enumerate(articles):
            try:
                pmid = article["MedlineCitation"]["PMID"]
                title = article["MedlineCitation"]["Article"]["ArticleTitle"]
                abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", ["No abstract available"])[0]
                
                # Sanitize filename
                safe_title = "".join(c if c.isalnum() else "_" for c in title)[:100]
                abstract_path = f"Parkinson_Resources/PubMed_Articles/Abstracts/{safe_title}.txt"
                
                # Save abstract
                with open(abstract_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\nPMID: {pmid}\n\nAbstract:\n{abstract}")
                logging.info(f"Saved abstract: {abstract_path} ({i+1}/{len(articles)})")

                # Attempt PDF download (if a PMC ID exists)
                pmc_id = None
                if "ArticleIdList" in article["PubmedData"]:
                    pmc_id = next((id for id in article["PubmedData"]["ArticleIdList"] if id.startswith("PMC")), None)

                if pmc_id:
                    self_hosted_pdf = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
                    # Use pmid for the API URL
                    api_pdf = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&id={pmid}&cmd=prlinks"

                    for pdf_url in [self_hosted_pdf, api_pdf]:
                        try:
                            response = requests.head(pdf_url, allow_redirects=True)
                            if response.status_code == 200:
                                pdf_response = requests.get(pdf_url)
                                pdf_path = f"Parkinson_Resources/PubMed_Articles/PDFs/{safe_title}.pdf"
                                with open(pdf_path, "wb") as f:
                                    f.write(pdf_response.content)
                                logging.info(f"Downloaded PDF via {pdf_url}")
                                break
                        except Exception as e:
                            continue

                time.sleep(1)  # Respect rate limits

            except Exception as e:
                logging.error(f"Error processing article {i+1}: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"PubMed fetch failed: {str(e)}")

def download_guidelines():
    """Download clinical guidelines with headers and retries"""
    guideline_urls = {
        "NICE": {
            "url": "https://www.nice.org.uk/guidance/ng71",
            "filename": "NICE_Parkinsons_Guidelines.pdf"
        },
        "MJFF": {
            "url": "https://www.michaeljfox.org/sites/default/files/media/document/Guide_For_Newly_Diagnosed_120921.pdf",
            "filename": "MJFF_Parkinsons_Overview.pdf"
        }
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }

    for source, data in guideline_urls.items():
        try:
            for attempt in range(3):
                try:
                    response = requests.get(data["url"], headers=headers, stream=True, timeout=10)
                    response.raise_for_status()
                    
                    # Check content type
                    if "pdf" not in response.headers.get("Content-Type", "").lower():
                        logging.warning(f"Unexpected content type for {source}: {response.headers.get('Content-Type')}")
                    
                    file_path = f"Parkinson_Resources/Clinical_Guidelines/{source}/{data['filename']}"
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            
                    if os.path.getsize(file_path) > 1024:  # At least 1KB
                        logging.info(f"Successfully downloaded {source} guidelines")
                        break
                    else:
                        os.remove(file_path)
                        logging.warning(f"Empty file downloaded on attempt {attempt+1} for {source}")
                        
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Attempt {attempt+1} failed for {source}: {str(e)}")
                    time.sleep(2**attempt)  # Exponential backoff
                    
        except Exception as e:
            logging.error(f"Final failure for {source}: {str(e)}")

if __name__ == "__main__":
    try:
        create_folders()  # Create required directories
        
        # PubMed search parameters
        search_term = (
            '(Parkinson\'s disease[Title/Abstract]) '
            'AND (review[Publication Type] OR journal article[Publication Type]) '
            'AND english[Language] '
            'AND ("2018/01/01"[Date - Publication] : "2023/12/31"[Date - Publication])'
        )
        fetch_pubmed_articles(search_term, max_articles=10)
        
        # Download clinical guidelines
        download_guidelines()
        
    except Exception as e:
        logging.error(f"Critical failure: {str(e)}")
