#pragma once

#include<iostream>
#include<fstream>
#include<sstream>
#include<tuple>
#include<map>
#include<math.h>
#include<string>
#include<vector>
using namespace std;

class D_star
{
public:
	explicit D_star();
	map<tuple<int,int,int>, double> Alldirec;   // 27个方向，获得子节点
	map<tuple<int, int, int>, tuple<int, int, int>> b;  // 记录反向指针，即从一个节点指向父节点，D*是反着指的
	map<tuple<int, int, int>, double> OPEN; // 开节点集
	map <tuple<int, int, int>, double> h;   // 路径耗散记录表
	map <tuple<int, int, int>, string> tag; // 标志：Open Closed New
	vector<tuple<int, int, int>> path;      // 记录规划路径的容器
	int count;                              // 记录迭代次数
	
	vector<vector<int>> obs_trace;

	tuple<int, int, int> start;             // 起始点位置
	tuple<int, int, int> goal;              // 终止点位置

	double cost(tuple<int, int, int>&, tuple<int, int, int>&); // 欧式距离计算函数（碰撞返回一个 大数）
	void check_state(tuple<int, int, int>&);                   // 添加信息
	double get_kmin();                                         // 获取开节点集最小值
	tuple<tuple<int, int, int>,double> min_state();            // 获取开节点集最小值并弹出元素
	void insert(tuple<int, int, int>&,double&);                // 插入开节点集并更新h表
	double process_state();                                    // D*核心算法 详见原论文
	vector<tuple<int, int, int>> children(tuple<int, int, int>&);    // 获取子节点
	void modify_cost(tuple<int, int, int>&);                         // 更新动态环境下节点信息
	void modify(tuple<int, int, int>&);                              // 同上，配合使用
	void get_path();                                                 // 获取规划路径
	void run();                                                      // 主函数 
	void save_path(string);                                          // 保存路径到csv
	void load_csv(string);
	void save_obs_trace(string);

	/*运动障碍球信息*/
	double obs_r;
	tuple<int, int, int> obs_pos;
};