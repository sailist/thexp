"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.

    规约：
    所有带有数据的传输一律使用post请求且以json形式传输（而不是form），同时定义以下几种数据字段

    request.json['exps'] list, 每一个元素表示一个 exp 信息，格式为 {exp_name, project_iname}
    request.json['data'] : list, 每一个元素表示一个 test 的 json_info
    request.json['test_name'] str, 表示 test 的唯一 name

    另外还有仅在某方法中使用的参数
    request.json['include_hide'] :bool, 表示是否包含隐藏文件
"""
import os
from flask import Flask, jsonify, request, redirect,send_file,make_response
from urllib.parse import quote,unquote

from .response import Responses

# app = Flask(__name__)
app = Flask(__name__, static_folder="dist", static_url_path="/", template_folder="dist")

res_call = None  # type:Responses


@app.route('/')
def index():
    return redirect('/index.html')


@app.route('/api/info/exps')
def exp_info():
    """获取 exp 级别的目录树，主页左上角数据"""
    return res_call.info_exps()


@app.route('/api/info/tests', methods=['POST'])
def test_info():
    """获取某些 experiments 对应的 test"""
    return res_call.info_tests(request.json['exps'], request.json['include_hide'])


@app.route('/api/info/test', methods=['POST'])
def one_test_info():
    """获取一个 test 对应的详细信息"""
    return res_call.info_test(request.json['test_name'])


@app.route('/api/base/layout', methods=['GET'])
def frontend_layout():
    """
    获取前端可控制部份的界面信息
    目前可控制的部份有：
        stat 部份，即表格上方的部份按钮区域，目前实现有 Base, Compare, ... Summary 共四个方法
    :return:
    """
    return res_call.base_layout()


@app.route('/api/base/refresh', methods=['GET'])
def refresh_viewer():
    """刷新 viewer，对应右上角刷新按钮，用于试验有变动但是前端没有更新的时候"""
    res_call.refresh()
    return jsonify(code=200)


@app.route('/api/op/<method>', methods=['POST'])
def test_visible(method):
    """
    控制试验正常情况下是否可见
    :param method: hide / show
    :return:
    """
    results, _ = res_call._base_analyse(request.json['data'])
    if method == 'hide':
        [i.hide() for i in results]
        return jsonify(code=200)
    elif method == 'show':
        [i.show() for i in results]
        return jsonify(code=200)
    elif method == 'fav':
        [i.fav() for i in results]
        return jsonify(code=200)
    elif method == 'unfav':
        [i.unfav() for i in results]
        return jsonify(code=200)


@app.route('/api/stat/<method>', methods=['POST'])
def stat_tests(method):
    """
    主页一次对卡片显示内容的请求，包含多个试验对象。
    :param method:  目前支持 base / summary / plot / compare ,
        该部份由 Response 对象中的 stat_{method} 动态定义和扩充，你可以按照该命名格式实现其他的统计类方法
        返回格式参考说明 TODO
    :return:
    """
    data = request.json['data']
    func = res_call.get_method("stat", method)
    if callable(func):
        return func(data)
    else:
        return jsonify(code=500, msg='未定义的方法: {}'.format(method))


@app.route('/api/git/<method>', methods=['POST'])
def git_op(method):
    """
    对某次 test 进行打包，或将本地文件恢复到某次 test 的状态
    :param method: 目前支持  reset / archive
    :return:
    """
    test_name = request.json['test_name']
    func = res_call.get_method('git', method)
    if callable(func):
        return func(test_name)


@app.route('/api/download/file',methods=['GET'])
def download_file():
    # fn = request.json['fn']
    fn = request.args.get('fn',None)
    if fn is None:
        return jsonify(code=500)
    fn = unquote(fn)
    if fn.endswith('.log'):
        if os.path.exists(fn):
            res = make_response(send_file(fn,as_attachment=True,attachment_filename=os.path.basename(fn)))
            res.headers['filename'] = quote(os.path.basename(fn))
            return res
        else:
            return jsonify(code=200,msg='文件不存在（可能已被删除）')
    else:
        return jsonify(code=500,msg='不支持的文件类型（目前仅支持 .log 后缀）')


def run(response=None, *args, **kwargs):
    """
    运行服务

    Args:
        response:
        *args:
        **kwargs:

    Returns:

    """
    global res_call
    if response is None:
        res_call = Responses()
    else:
        res_call = response

    import os
    # os.environ['FLASK_ENV'] = "production"
    from .util import get_ip
    if kwargs.get('host',None) is None:
        kwargs['host'] = get_ip()
    if kwargs.get('port',None) is None:
        kwargs['port'] = '5555'

    app.run(debug=False, *args, **kwargs)
