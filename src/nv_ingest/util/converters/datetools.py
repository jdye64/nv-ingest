# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime, timedelta

from dateutil.parser import parse

from nv_ingest.util.exception_handlers.converters import datetools_exception_handler


@datetools_exception_handler
def datetimefrompdfmeta(pdf_formated_date, keep_tz=False):
    """
    Convert PDF metadata formated datestring to datetime object

    Parameters
    ----------
    pdf_formated_date : str
        A date string in standard PDF metadata format.
        Example: `str("D:20211222141131-07'00'")`
    keep_tz : bool
        Keep or remove timezone attribute of parsed datetime object.
        If `False`, (necessary for arrow format) timezone offset will be added to datetime.
        Parsed datetimes will be in the same local time.

    Returns
    -------
    datetime.datetime
        A datetime object parsed from the input date string.

    """

    try:
        # standard pdf date format
        pattern = "D:%Y%m%d%H%M%S%z"
        # clean up date string
        cleaned_date_string = pdf_formated_date[:-1].replace("'", ":")
        parsed_dt_tz = datetime.strptime(cleaned_date_string, pattern)
    except ValueError:
        parsed_dt_tz = parse(pdf_formated_date, fuzzy=True)

    if not keep_tz:
        return remove_tz(parsed_dt_tz).isoformat()

    return parsed_dt_tz.isoformat()


def remove_tz(datetime_obj):
    """
    Remove timezone and add offset to datetime object.

    Parameters
    ----------
    datetime_obj : datetime.datetime
        A datetime object with or without timezone attribute set.

    Returns
    -------
    datetime.datetime
        A datetime object with timezone offset added and timezone attribute removed.

    """

    if datetime_obj.tzinfo is None:
        return datetime_obj

    utc_delta = timedelta(seconds=datetime_obj.utcoffset().total_seconds())

    datetime_obj = datetime_obj.replace(tzinfo=None)

    return datetime_obj + utc_delta


def validate_iso8601(date_string):
    """
    Verify date string is in ISO 8601 format.

    parameters
    ----------
    date_string : str
        A date in human readable format, ideally ISO 8601.
    """

    assert datetime.fromisoformat(date_string)
