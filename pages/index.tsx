import type { NextPage } from 'next'
import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'
import { useRouter } from 'next/router'

const Home: NextPage = () => {
  const router = useRouter()

  const onPlaygroundClick = () => {
    router.push({ // tell NEXT.js to redirct the page
      pathname: '/playground/'
    })
  }

  return (
    <div className={styles.container}>
      <Head>
        <title>MLIR-China Community</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/mlir.png" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>
          欢迎来到
          <span className={styles.logo}>
            <Image src="/mlir.png" alt="MLIR Logo" width={60} height={60} />
          </span>
          <a href="https://mlir.llvm.org/">MLIR</a> 中国社区！
        </h1>

        <p className={styles.description}>
          迎接后摩尔时代，加速中国普惠新硬件时代的到来🚀🚀🚀
        </p>

        <div className={styles.grid}>
          <a href="mlir-china.org" className={styles.card}>
            <h2>编译器原理课程 &rarr;</h2>
            <p>基于MLIR的编译器原理课程(开发中...)</p>
          </a>

          <a onClick={onPlaygroundClick} className={styles.card}>
            <h2>Playground &rarr;</h2>
            <p>在浏览器中体验使用MLIR进行编译器逻辑的开发!</p>
          </a>

          <a href="https://www.zhihu.com/people/mlir-70" className={styles.card}>
            <h2>知乎社区 &rarr;</h2>
            <p>社区文章首发平台，敬请关注和投稿！</p>
          </a>

          <a href="https://github.com/MLIR-China" className={styles.card}>
            <h2>Github社区 &rarr;</h2>
            <p>社区代码开发平台，欢迎参与各类工具的开发！</p>
          </a>

        </div>
      </main>

    </div>
  )
}

export default Home
