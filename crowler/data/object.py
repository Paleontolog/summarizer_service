from scrapy import Item, Field


class Step(Item):
    number = Field()
    abstract = Field()
    text = Field()


class Variant(Item):
    variant_name = Field()
    steps = Field()


class Article(Item):
    main_title = Field()
    main_description = Field()
    variants = Field()
    advices = Field()
    warnings = Field()
    link = Field()
