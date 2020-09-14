# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    reviewTitle = scrapy.Field()  # 제목
    reviewText = scrapy.Field()  # 리뷰
    pass
