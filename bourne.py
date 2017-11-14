"""
Copyright (c) 2017 Douglas Farinelli

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import abc
import argparse
import asyncio
import datetime as datetimepy
import enum
import functools
import importlib
import inspect
import os

from typing import Callable, Dict, Iterable, List, Tuple, Union


DEFAULT_DATE_FORMAT = os.getenv('BOURNE_DEFAULT_DATE_FORMAT', '%Y-%m-%d')

DEFAULT_TIME_FORMAT = os.getenv('BOURNE_DEFAULT_TIME_FORMAT', '%H:%M:%S')


def _normalize_hyphens(param: str) -> str:
    if param.startswith('-'):
        param = param[1:]
        param = _normalize_hyphens(param)
    return param


class ValidationError(argparse.ArgumentTypeError):
    pass


class Parameter:

    default_validators = []

    def __init__(
        self,
        extended: str,
        name: str,
        alias: str=None,
        help: str=None,
        default=None,
        choices: Union[enum.Enum, Iterable]=None,
        required: bool=False,
        many: bool=False,
        to_python: Callable=None,
        validators: Iterable[Callable]=None,
    ) -> None:

        self.alias = None
        if alias:
            self.alias = f'-{_normalize_hyphens(alias)}'

        self.is_enum = (
            inspect.isclass(choices)
            and issubclass(choices, enum.Enum)
        )

        self.choices = choices

        self.command = None
        self.default = default
        self.help = help
        self.name = name

        self.extended = f'--{_normalize_hyphens(extended)}'

        self.many = many
        self.required = required
        self.validators = self.default_validators[:]

        if validators:
            self.validators.extend(validators)

        self.to_python = to_python

    def clean(self, value):
        """
        Validate the given value and return its "cleaned" value as an
        appropriate Python object. Raise ValidationError for any errors.
        """
        to_python = self.to_python
        self.run_validators(value)
        return to_python(value) if to_python else value

    def run_validators(self, value) -> None:
        for validator in self.validators:
            validator(value)

    def add_to_parser(self, parser: argparse.ArgumentParser):
        args, kwargs = self._prepare_arguments_to_add_parser()
        parser.add_argument(*args, **kwargs)

    @classmethod
    def as_decorator(cls):
        def decorator(*args, **kwargs):
            def _define_to_command(command):
                command = Command.create_from_callable(command)
                command.add_parameter(cls(*args, **kwargs))
                return command
            return _define_to_command
        return decorator

    @classmethod
    def as_management_method(cls):
        def method(self, *args, **kwargs):
            self.add_parameter(cls(*args, **kwargs))
        return method

    def _prepare_arguments_to_add_parser(self) -> Tuple:
        args = self._get_name_or_flags_argument_parser()
        kwargs = dict(
            action=self._get_action_argument_parser(),
            choices=self._get_choices_argument_parser(),
            default=self._get_default_argument_parser(),
            dest=self._get_dest_argument_parser(),
            help=self._get_help_argument_parser(),
            nargs=self._get_nargs_argument_parser(),
            required=self._get_required_argument_parser(),
            type=self._get_type_argument_parser(),
        )
        return args, kwargs

    def _get_name_or_flags_argument_parser(self) -> Tuple:
        if self.alias:
            return self.alias, self.extended
        return self.extended,

    def _get_action_argument_parser(self):
        pass

    def _get_choices_argument_parser(self) -> Union[enum.Enum, Iterable, None]:  # NOQA
        return self.choices

    def _get_dest_argument_parser(self) -> str:
        return self.name

    def _get_default_argument_parser(self):
        return self.default

    def _get_help_argument_parser(self):
        return self.help

    def _get_required_argument_parser(self) -> bool:
        return self.required

    def _get_nargs_argument_parser(self) -> Union[str, None]:
        if self.many:
            return '+' if self.required else '*'
        return None

    def _get_type_argument_parser(self) -> Callable:
        return self.clean


class Boolean(Parameter):

    def __init__(
        self,
        extended: str,
        name: str,
        alias: str=None,
        help: str=None,
        validators: Iterable[Callable]=None,
    ):
        super(Boolean, self).__init__(
            alias=alias,
            choices=None,
            default=False,
            extended=extended,
            help=help,
            many=False,
            name=name,
            required=False,
            to_python=None,
            validators=validators,
        )

    def _get_action_argument_parser(self) -> str:
        return 'store_true'

    def _prepare_arguments_to_add_parser(self) -> Tuple:
        args, kwargs = super(Boolean, self)._prepare_arguments_to_add_parser()
        kwargs.pop('nargs', None)
        kwargs.pop('type', None)
        kwargs.pop('choices', None)
        return args, kwargs


class Many(Parameter):

    def __init__(
        self,
        extended: str,
        name: str,
        alias: str=None,
        help: str=None,
        default=None,
        choices=None,
        required=False,
        to_python=None,
        validators: Iterable[Callable]=None,
    ):
        super(Many, self).__init__(
            alias=alias,
            choices=choices,
            default=default,
            extended=extended,
            help=help,
            many=True,
            name=name,
            required=required,
            to_python=to_python,
            validators=validators,
        )


class DateTime(Parameter):

    format = None

    _check_default_instance = datetimepy.datetime

    def __init__(
        self,
        extended: str,
        name: str,
        alias: str=None,
        help: str=None,
        default=None,
        choices: Union[enum.Enum, Iterable]=None,
        required: bool=False,
        many: bool=False,
        validators: Iterable[Callable]=None,
        format: str=None,
    ):

        if default and not isinstance(default, self._check_default_instance):
            raise RuntimeError(
                f'The default must be a {self._check_default_instance} '
                f'instance'
            )

        self.format = (
            format
            or self.format
            or f'{DEFAULT_DATE_FORMAT} {DEFAULT_TIME_FORMAT}'
        )

        super(DateTime, self).__init__(
            alias=alias,
            choices=choices,
            default=default,
            extended=extended,
            help=help,
            many=many,
            name=name,
            required=required,
            to_python=self.to_python,
            validators=validators,
        )

    def to_python(self, value):
        if isinstance(value, datetimepy.datetime):
            return value
        return datetimepy.datetime.strptime(value, self.format)

    def clean(self, value):
        try:
            return super(DateTime, self).clean(value)
        except ValueError as e:
            raise ValidationError(
                f'Invalid {self.__class__.__name__.lower()} "{value}", {e}'
            ) from e


class Date(DateTime):

    format = DEFAULT_DATE_FORMAT

    _check_default_instance = datetimepy.date

    def to_python(self, value):
        value = super(Date, self).to_python(value)
        return value.date()


class Time(DateTime):

    format = DEFAULT_TIME_FORMAT

    _check_default_instance = datetimepy.time

    def to_python(self, value):
        value = super(Time, self).to_python(value)
        return value.time()


class Command(metaclass=abc.ABCMeta):

    name = None

    help = None

    parent = None

    def __init__(
        self,
        name: str=None,
        help: str=None,
        parent: argparse._SubParsersAction=None
    ) -> None:
        self.name = name or self.name
        self.help = help or self.help
        self.__parameters = {}  # type: Dict[str, Parameter]
        self.parent = parent or self.parent or management.commands

        self.parser = self.parent.add_parser(
            self.name,
            help=self.help
        )

        self.parser.set_defaults(handler=self)

        self.subparser = self.parser.add_subparsers()

        self.__subcommands = {}  # type: Dict[str, Command]

        self.prepare()

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return self.__parameters

    @property
    def subcommands(self) -> Dict[str, 'Command']:
        return self.__subcommands

    @abc.abstractmethod
    def __call__(self, options: argparse.Namespace) -> None:
        pass

    def prepare(self) -> None:
        pass

    def add_parameter(self, parameter: Parameter) -> None:
        parameter.command = self
        parameter.add_to_parser(self.parser)
        self.__parameters[parameter.name] = parameter

    def add_subcommand(self, command: 'Command') -> None:
        self.__subcommands[command.name] = command

    def subcommand( # TODO: criar classe
        self,
        callable: Callable,
        name: str=None,
        help: str=None
    ) -> 'Command':
        command = self.create_from_callable(
            callable=callable,
            name=name,
            help=help,
            parent=self.subparser
        )
        self.add_subcommand(command)
        return command

    @classmethod
    def as_decorator(cls):
        return functools.partial(
            cls.create_from_callable,
            name=None,
            help=None,
            parent=None,
        )

    @classmethod
    def create_from_callable(
        cls,
        callable: Callable,
        name: str=None,
        help: str=None,
        parent: argparse._SubParsersAction=None
    ) -> 'Command':
        if isinstance(callable, cls):
            return callable

        doc = help or callable.__doc__

        name = name or callable.__name__

        klass = type(name, (cls,), {
            'name': name,
            'help': doc,
            'parent': parent,
            '__call__': staticmethod(callable),
        })

        functools.update_wrapper(wrapper=klass, wrapped=callable, updated=[])

        return klass()


class Management:

    def __init__(self) -> None:
        self.cli = argparse.ArgumentParser()
        self.commands = self.cli.add_subparsers()
        self.__parameters = {}  # type: Dict[str, Parameter]

    def main(self, argv=None) -> None:
        options = self.cli.parse_args(args=argv)

        handler = getattr(options, 'handler', None)

        if handler is None:
            return self.cli.print_help()

        start = datetimepy.datetime.utcnow()

        try:
            response = handler(options)

            if asyncio.iscoroutine(response):

                loop = asyncio.get_event_loop()
                loop.run_until_complete(response)
        finally:
            end = datetimepy.datetime.utcnow()
            print(f'\nExecuted in {end - start}.')

    parameter = Parameter.as_management_method()

    boolean = Boolean.as_management_method()

    many = Many.as_management_method()

    date = Date.as_management_method()

    datetime = DateTime.as_management_method()

    time = Time.as_management_method()

    def add_parameter(self, parameter: Parameter) -> None:
        parameter.add_to_parser(self.cli)
        self.__parameters[parameter.name] = parameter

    @staticmethod
    def load_from_modules(modules: List[str]) -> None:
        for dotted_path in modules:
            importlib.import_module(dotted_path)


management = Management()

main = management.main

command = Command.as_decorator()

parameter = Parameter.as_decorator()

boolean = Boolean.as_decorator()

many = Many.as_decorator()

date = Date.as_decorator()

datetime = DateTime.as_decorator()

time = Time.as_decorator()
