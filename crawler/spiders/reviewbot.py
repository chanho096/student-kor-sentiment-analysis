import scrapy
import os
from crawler.items import CrawlerItem

item = CrawlerItem()

class ReviewSpider(scrapy.Spider):
    name = os.path.basename(__file__)
    fileRangeName = "plandas"
    base_url = "https://movie.daum.net/moviedb/main?movieId=1"

    def __init__(self, domain=''):
        self.base_url = str(domain)

    # read data with moving urls
    def start_requests(self):
        url_part1 = self.base_url.replace("main", "grade")
        for reviewIndex in range(1, 10):
            yield scrapy.Request("{0}&page={1}".format(url_part1, reviewIndex),
            callback=self.parse_review_n_rank)

    # read datas in one movie review
    def parse_review_n_rank(self, response):
        # is there review
        if len(response.xpath('//*[@id="mArticle"]/div[2]/div[2]/div[1]/p')) == 0:
            numOfli = len(response.xpath('//*[@id="mArticle"]/div[2]/div[2]/div[1]/ul/li').extract())
            for i in range(1, numOfli):
                reviewText = response.xpath('//*[@id="mArticle"]/div[2]/div[2]/div[1]/ul/li[{0}]/div/p/text()'.format(i))
                reviewGrade = response.xpath('//*[@id="mArticle"]/div[2]/div[2]/div[1]/ul/li[{0}]/div/div[1]/em/text()'.format(i))
                reviewTitle = response.xpath('//*[@id="mArticle"]/div[1]/a/h2/text()')

                item['reviewTitle'] = reviewTitle.extract()
                item['reviewText'] = reviewText.extract()
                item['reviewGrade'] = reviewGrade.extract()

                yield item
