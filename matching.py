import numpy

from product import Product
import jellyfish
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def __monge_elkan_token_similarity(invoice_token: str, product_tokens: list[str]):
    """ Calculates the maximum similarity between the invoice token and all tokens of the product name.

        Similarity calculation can be done using any similarity or distance metric.
        This version will use Jaro-Winkler to determine similarity

    Parameters
    ----------
    :param invoice_token: token to calculate similarity for
    :param product_tokens: tokens of the product name
    :return: maximum similarity of invoice token with any product token: float
    """

    max_similarity = float("-inf")
    for token in product_tokens:
        similarity = jellyfish.jaro_winkler_similarity(invoice_token, token)

        if similarity > max_similarity:
            max_similarity = similarity

    return max_similarity


def __term_frequency(token_set: list[str], document: list[str]):
    """ Calculates the term frequency for each token in the token set based on the document

    Parameters
    ----------
    :param token_set: tokens for which the term frequency shall be calculated
    :param document: document for term frequency calculation
    :return: list of term frequencies: list[float]
    """
    term_freq = []

    for token in token_set:
        term_freq.append((float(document.count(token)) / len(document)))

    return term_freq


def __inverse_document_frequency(token_set: list[str], corpus: list[list[str]]):
    """ Calculates the inverse document frequency for each token based on the corpus

    Parameters
    ----------
    :param token_set: tokens for which the inverse document frequency shall be calculated
    :param corpus: all documents for inverse document frequency calculation
    :return: list of normalized inverse document frequencies: list[float]
    """

    inverse_document_freq = []

    for token in token_set:
        idf = math.log((len(corpus)/sum(token in x for x in corpus)))
        inverse_document_freq.append(idf)

    normalized_inverse_document_freq = []
    normalization = math.sqrt(pow(sum(x for x in inverse_document_freq), 2))

    for idf in inverse_document_freq:
        normalized_inverse_document_freq.append(idf/normalization)

    return normalized_inverse_document_freq


def __tf_idf(term_frequency: list[float], inverse_document_frequency):
    tfidf = []
    for index in range(len(term_frequency)):
        tfidf.append(term_frequency[index] * inverse_document_frequency[index])

    return tfidf


def levenshtein_matcher(reference_products: list[Product], invoice):
    """ Calculates the best possible match based on Levenshtein Distance

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: product with the lowest Levenshtein Distance: tuple[Product, int]
    """

    shortest_distance = float("inf")
    match_product = None

    for product in reference_products:
        distance = jellyfish.levenshtein_distance(product.name, invoice.name)

        if distance < shortest_distance:
            shortest_distance = distance
            match_product = product

    return tuple((match_product, shortest_distance))


def jaro_matcher(reference_products: list[Product], invoice, use_prefix: bool = False):
    """ Calculates the Jaro similarity between the invoice and all products

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice:  extracted line of invoice
    :param use_prefix: determines whether Jaro or Jaro-Winkler is used; default = False
    :return:  product with the highest similarity: tuple[Product, float]
    """

    best_match = float("-inf")
    match_product = Product("", "", "")

    for product in reference_products:
        if use_prefix:
            match = jellyfish.jaro_winkler_similarity(product.name, invoice.name)
        else:
            match = jellyfish.jaro_similarity(product.name, invoice.name)

        if match > best_match:
            best_match = match
            match_product = product

    return tuple((match_product, best_match))


def tfidf_matcher(reference_products: list[Product], invoice):
    """ Finds the most similar product based on TF-IDF and cosine similarity

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: products with the highest similarity: tuple(list[Product], float)
    """

    documents = [invoice.name]
    documents.extend([x.name for x in reference_products])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    all_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    similarities = all_similarities[0:1, 1::1]
    max_similarity = numpy.amax(similarities)
    index = numpy.where(similarities == max_similarity)

    similar_products = [reference_products[x] for x in index[1]]

    return tuple((similar_products, max_similarity))


def jaccard_matcher(reference_products: list[Product], invoice):
    """ Finds the most similar product using Jaccard similarity

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: product with the highest similarity: tuple(Product, float)
    """

    # tokenize all strings
    invoice_tokens = invoice.name.split()
    product_tokens = [x.name.split() for x in reference_products]

    max_similarity = float("-inf")
    match_product = None

    for index, product in enumerate(product_tokens):
        intersection = len(set(invoice_tokens).intersection(product))
        union = len(set(invoice_tokens).union(product))
        jaccard_similarity = float(intersection) / union

        if jaccard_similarity > max_similarity:
            max_similarity = jaccard_similarity
            match_product = reference_products[index]

    return tuple((match_product, max_similarity))


def monge_elkan_matcher(reference_products: list[Product], invoice):
    """ Finds the most similar product using Monge-Elkan strategy

        Monge-Elkan is a hybrid similarity metric that utilizes tokens and calculates the similarity of two strings
        using the similarity of the individual tokens.

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: product with the highest similarity: tuple(Product, float)
    """

    # tokenize strings
    invoice_tokens = invoice.name.split()
    num_invoice_tokens = len(invoice_tokens)
    product_tokens = [x.name.split() for x in reference_products]

    max_similarity = float("-inf")
    match_product = None

    for index, product in enumerate(product_tokens):
        sum_token_similarity = 0.0

        for token in invoice_tokens:
            sum_token_similarity += __monge_elkan_token_similarity(token, product)

        monge_elkan = (1 / num_invoice_tokens) * sum_token_similarity

        if monge_elkan > max_similarity:
            max_similarity = monge_elkan
            match_product = reference_products[index]

    return tuple((match_product, max_similarity))


def soft_tfidf_matcher(reference_products: list[Product], invoice):
    """ Finds the most similar match based on a version of TF-IDF that not only considers exact matches.
        Jaro-Winkler is used as an additional similarity metric for token selection

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoice: extracted line of invoice
    :return: product with the highest similarity: tuple(Product, float)
    """

    # tokenize strings
    invoice_tokens = invoice.name.split()
    product_tokens = [x.name.split() for x in reference_products]
    corpus = product_tokens
    corpus.append(invoice_tokens)

    # tf = number of times a token appears in the document/string
    # idf = weight of words across the whole corpus  (log(number of documents / number of documents containing token)
    # tf-idf = tf * idf
    # similarity = (tfidf * tfidf * similarity) + (...)

    best_similarity = 0.0
    best_product = None

    for product_index, product in enumerate(product_tokens):
        invoice_similarity_tokens = []
        product_similarity_tokens = []
        jaro_winkler_similarities = []

        # select tokens usable for matching using Jaro-Winkler
        for invoice_token in invoice_tokens:
            best_jaro_metric = 0.0
            best_token = ""
            for product_token in product:
                metric = jellyfish.jaro_winkler_similarity(invoice_token, product_token)

                if metric > best_jaro_metric:
                    best_jaro_metric = metric
                    best_token = product_token

            if best_jaro_metric > 0.55:
                invoice_similarity_tokens.append(invoice_token)
                product_similarity_tokens.append(best_token)
                jaro_winkler_similarities.append(best_jaro_metric)

        # calculate term frequency of tokens
        invoice_term_frequency = __term_frequency(invoice_similarity_tokens, invoice_tokens)
        product_term_frequency = __term_frequency(product_similarity_tokens, product)

        # calculate inverse document frequency of tokens
        invoice_inverse_document_freq = __inverse_document_frequency(invoice_similarity_tokens, corpus)
        product_inverse_document_freq = __inverse_document_frequency(product_similarity_tokens, corpus)

        # calculate tf-idf
        invoice_tfidf = __tf_idf(invoice_term_frequency, invoice_inverse_document_freq)
        product_tfidf = __tf_idf(product_term_frequency, product_inverse_document_freq)

        # calculate similarity
        tmp_similarity = 0.0
        for idx in range(len(invoice_tfidf)):
            tmp_similarity += invoice_tfidf[idx] * product_tfidf[idx] * jaro_winkler_similarities[idx]

        if tmp_similarity > best_similarity:
            best_similarity = tmp_similarity
            best_product = reference_products[product_index]

    return tuple((best_product, best_similarity))


def levenshtein_bulk_matcher(reference_products: list[Product], invoices: list):
    """ Finds the product with the smallest Levenshtein distance for each invoice items

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoices: list of invoice items to find a match for
    :return: list of best matches
    """

    matches = []

    for invoice in invoices:
        matches.append(levenshtein_matcher(reference_products, invoice))

    return matches


def jaro_bulk_matcher(reference_products: list[Product], invoices: list):
    """ Finds the product with the highest Jaro similarity for each invoice items

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoices: list of invoice items to find a match for
    :return: list of best matches
    """

    matches = []

    for invoice in invoices:
        matches.append(jaro_matcher(reference_products, invoice))

    return matches


def jaro_winkler_bulk_matcher(reference_products: list[Product], invoices: list):
    """ Finds the product with the highest Jaro-Winkler similarity for each invoice items

    Parameters
    ----------
    :param reference_products: products to match the invoice object against
    :param invoices: list of invoice items to find a match for
    :return: list of best matches
    """

    matches = []

    for invoice in invoices:
        matches.append(jaro_matcher(reference_products, invoice, True))

    return matches


def jaccard_bulk_matcher(reference_products: list[Product], invoices: list):
    """ Finds the product with the highest Jaccard similarity for each invoice items

        Parameters
        ----------
        :param reference_products: products to match the invoice object against
        :param invoices: list of invoice items to find a match for
        :return: list of best matches
        """

    matches = []

    for invoice in invoices:
        matches.append(jaccard_matcher(reference_products, invoice))

    return matches


def monge_elkan_bulk_matcher(reference_products: list[Product], invoices: list):
    """ Finds the product with the highest Monge-Elkan similarity for each invoice items

            Parameters
            ----------
            :param reference_products: products to match the invoice object against
            :param invoices: list of invoice items to find a match for
            :return: list of best matches
            """

    matches = []

    for invoice in invoices:
        matches.append(monge_elkan_matcher(reference_products, invoice))

    return matches


def tfidf_bulk_matcher(reference_products: list[Product], invoices: list):
    """ Finds the product with the highest tf_idf cosine similarity for each invoice items

            Parameters
            ----------
            :param reference_products: products to match the invoice object against
            :param invoices: list of invoice items to find a match for
            :return: list of best matches
            """

    matches = []

    for invoice in invoices:
        matches.append(tfidf_matcher(reference_products, invoice))

    return matches


def soft_tfidf_bulk_batcher(reference_products: list[Product], invoices: list):
    """ Finds the product with the highest soft (2-level) TF-IDF similarity for each invoice items

            Parameters
            ----------
            :param reference_products: products to match the invoice object against
            :param invoices: list of invoice items to find a match for
            :return: list of best matches
            """

    matches = []

    for invoice in invoices:
        matches.append(soft_tfidf_matcher(reference_products, invoice))

    return matches
