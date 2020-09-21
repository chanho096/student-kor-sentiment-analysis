import scrapy
import os
from crawler.items import CrawlerItem
from crawler.utils import MovieCrawler

item = CrawlerItem()
utils = MovieCrawler()

class ReviewSpider(scrapy.Spider):
    name = os.path.basename(__file__)
    base_url = "https://movie.daum.net/moviedb/main?movieId=1"
    current_page = 1
    handle_httpstatus_list = [500]

    def __init__(self, domain=''):
        self.base_url = str(domain)

    # read data with moving urls
    def start_requests(self):
        url_part1 = self.base_url.replace("main", "grade")
        yield scrapy.Request("{0}&page=1".format(url_part1),
        callback=self.parse_review_n_rank)

    # read datas in one movie review
    def parse_review_n_rank(self, response):
        # is there review
        if response.status not in self.handle_httpstatus_list:
            if len(response.xpath('//*[@id="mArticle"]/div[2]/div[2]/div[1]/p')) == 0:
                numOfli = len(response.xpath('//*[@id="mArticle"]/div[2]/div[2]/div[1]/ul/li').extract())
                for i in range(1, numOfli+1):
                    reviewText = response.xpath('//*[@id="mArticle"]/div[2]/div[2]/div[1]/ul/li[{0}]/div/p/text()'.format(i))
                    reviewTitle = response.xpath('//*[@id="mArticle"]/div[1]/a/h2/text()')
                    item['reviewTitle'] = reviewTitle.extract()
                    item['reviewText'] = reviewText.extract()
                    yield item

                url_part1 = self.base_url.replace("main", "grade")
                self.current_page = self.current_page+1
                yield scrapy.Request("{0}&page={1}".format(url_part1, str(self.current_page)),
                                      callback=self.parse_review_n_rank, dont_filter=True)
            else:
                utils.is_end = True
        else:
            utils.is_error = True