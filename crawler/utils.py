from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from crawler.pipelines import TextPipeline
import os

settings_file_path = 'crawler.settings'  # The path seen from root, ie. from main.py


class MovieCrawler:
    is_error = False

    def __init__(self, bot="reviewbot.py"):
        # 환경 변수 설정
        os.environ.setdefault('SCRAPY_SETTINGS_MODULE', settings_file_path)

        self.process = CrawlerProcess(get_project_settings())
        self.bot = bot

    def crawl(self, url):
        self.is_error = False
        self.process.crawl(self.bot, domain=url)
        self.process.start()  # the script will block here until the crawling is finished
        return TextPipeline.list_csv
