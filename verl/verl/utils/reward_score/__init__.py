# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from . import math_score

def _default_compute_score(data_source, solution_str, ground_truth, response_length=None, extra_info=None, is_test=False, is_check_correctness=True):
    return math_score.compute_score(solution_str, ground_truth, response_length, is_test=is_test, is_check_correctness=is_check_correctness)
