# Git 원격 저장소 활용

Git 원격 저장소 기능을 제공 해주는 서비스는 `gitlab`, `bitbucket`, `github` 등이 있다.

## 0. 원격 저장소 설정

``` bash
$ git remote add origin (url)
$ git remote add origin https://github.com/sehun4239/TIL.git
```

* git, 원격저장소를 추가(`add`)하고 origin 이라는 이름으로 `url`을 설정

* 설정된 저장소를 확인하기 위해서는 아래의 명령어를 사용한다.

  * ```bash
    git remote -v
    origin  https://github.com/sehun4239/TIL.git (fetch)
    origin  https://github.com/sehun4239/TIL.git (push)
    ```



## 원격 저장소에 `push`

```bash
$ git push origin master
Enumerating objects: 8, done.
Counting objects: 100% (8/8), done.
Delta compression using up to 8 threads
Compressing objects: 100% (5/5), done.
Writing objects: 100% (5/5), 877 bytes | 877.00 KiB/s, done.
Total 5 (delta 1), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
To https://github.com/sehun4239/TIL.git
   9527404..94ca05d  master -> master
```

