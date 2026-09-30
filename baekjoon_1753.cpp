#include<iostream>
#include<vector>
#include<queue>

using namespace std;

const int INF = 1e9;

int V, E;
int start;
int u, v, w;

int dis[20001];
vector<pair<int, int> > nodes[20001];
priority_queue<pair<int, int>, vector<pair<int, int> >, greater<pair<int, int> > > pq;

void dij() {

    dis[start] = 0;
    pq.push({0, start});

    while (!pq.empty()){
        int cur_dis = pq.top().first;
        int cur_node = pq.top().second;
        pq.pop();

        if (cur_dis > dis[cur_node]) continue;

        for (auto &edge : nodes[cur_node]){

            int node = edge.first;
            int new_dis = cur_dis + edge.second;

            if (new_dis < dis[node]) {
                dis[node] = new_dis;
                pq.push({dis[node], node});
            }
        }
    }
}



int main () {

    cin >> V >> E;

    cin >> start;

    for (int i = 1; i <= V; i++) {
        dis[i] = INF;
    }

    for (int i = 0; i < E; i++) {
        cin >> u >> v >> w;
        nodes[u].push_back(make_pair(v, w));
    }

    dij();

    for (int j = 1; j <= V; j++) {

        if (dis[j] != INF) {
            cout << dis[j] << '\n';
        } else {
            cout << "INF" << '\n';
        }
    }

}

// https://velog.io/@panghyuk/%EC%B5%9C%EB%8B%A8-%EA%B2%BD%EB%A1%9C-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98