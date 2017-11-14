import asyncio
import aiohttp
import bourne
import re

RE_ZIPCODE = re.compile(r'[0-9]{5}-[0-9]{3}')


def is_valid(value):
    if RE_ZIPCODE.match(value) is None:
        raise bourne.ValidationError(
            f'Invalid zipcode "{value}". Try this format: 09123-456.'
        )


@bourne.command
def zipcode():
    """Zipcode suite."""


@bourne.many(
    'codes',
    name='codes',
    required=True,
    validators=[is_valid]
)
@zipcode.subcommand
async def find(options):
    """
    Find one or more zip codes to find, ex.: --codes 09123-456
    """
    host = 'http://api.postmon.com.br/v1/cep/{}'

    session = aiohttp.ClientSession()

    with session:

        for address in await asyncio.gather(*[
            session.get(url=host.format(code)) for code in options.codes
        ]):

            json = await address.json()

            print(
                f'{json["cep"]:>10}',
                f'{json["logradouro"]:^60}',
                f'{json["cidade"]:^30}',
                f'{json["estado"]:>5}',
                sep=' | '
            )


if __name__ == '__main__':
    bourne.main()
