from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from crawler.pipelines import TextPipeline


class MovieCrawler:
    def __init__(self, bot="reviewbot.py"):
        self.process = CrawlerProcess(get_project_settings())
        self.bot = bot

    def crawl(self, url):
        self.process.crawl(self.bot, domain=url)
        self.process.start()  # the script will block here until the crawling is finished
        return TextPipeline.list_csv
